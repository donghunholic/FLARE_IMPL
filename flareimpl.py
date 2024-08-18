from dotenv import load_dotenv
import os

import bs4
from langchain import hub
from langchain.chains.llm import LLMChain
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader, WikipediaLoader, TextLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings, OpenAI
from langchain_core.prompts import PromptTemplate

from langchain_community.retrievers.bm25 import BM25Retriever
from NaverSearch import search_naver

load_dotenv()

question = "임진왜란은 몇년부터 몇년까지 일어났는지, 그리고 굵직한 사건들을 중심으로 설명해줘"

# llm = ChatOpenAI(model_name='gpt-3.5-turbo', temperature=0, max_tokens=4096)
#
# search_prompt = PromptTemplate.from_template(
#     """
#     질문을 대답하기 위해 검색해야할 핵심 키워드를 하나만 추출해주세요. (ex. #Question: 뮤지컬 해적의 캐스팅을 알려줘 #Query: 뮤지컬 해적)
#     #Question:
#     {question}
#     #Query:
#     """
# )
#
#
# search_chain = (
#     {"question": RunnablePassthrough()}
#     | search_prompt
#     | llm
#     | StrOutputParser()
# )
#
# search_query = search_chain.invoke(question)
#
# # 뉴스기사 내용을 로드하고, 청크로 나누고, 인덱싱합니다.
# loader = WebBaseLoader(
#     web_paths=search_naver(search_query),
#     bs_kwargs=dict(
#         parse_only=bs4.SoupStrainer("body"),
#     ),
#     verify_ssl=False,
# )

loader = TextLoader(
    file_path="./data/wiki.txt",
    autodetect_encoding=True
)

docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
splits = text_splitter.split_documents(docs)
print(f"분할 청크의 수: {len(splits)}")

retriever = BM25Retriever.from_documents(splits, k=3)

prompt = PromptTemplate.from_template(
    """당신은 질문-답변(Question-Answering)을 수행하는 친절한 AI 어시스턴트입니다. 당신의 임무는 주어진 문맥(context) 에서 주어진 질문(question)에 가능한 한 세부적인 내용까지 답변하세요. 만약, 주어진 문맥(context) 에서 답을 찾을 수 없다면, 답을 모른다면 `주어진 정보에서 질문에 대한 정보를 찾을 수 없습니다` 라고 답하세요.
모든 답변은 한글로 작성해 주세요. 단, 기술적인 용어나 이름은 번역하지 않고 그대로 사용해 주세요.

#Question: 
{question} 

#Context: 
{context} 

#Answer:"""
)

import langchain

langchain.verbose = True
from langchain.chains.flare.base import FlareChain, QuestionGeneratorChain, _OpenAIResponseChain

flare_llm = OpenAI(model_name="gpt-3.5-turbo-instruct", max_tokens=32, logprobs=1)

question_generator_chain = QuestionGeneratorChain(llm=flare_llm)
response_llm = OpenAI(model_name="gpt-3.5-turbo-instruct", max_tokens=32, logprobs=1)
response_chain = _OpenAIResponseChain(llm=response_llm)

flare = FlareChain(
    question_generator_chain=question_generator_chain,
    response_chain=response_chain,
    retriever=retriever,
    min_prob=0.2,
    min_token_gap=5,
    num_pad_tokens=2,
    max_iter=10,
    start_with_retrieval=True
)

# `flare.run()` 대신 `flare.invoke()` 사용
result = flare.invoke({"user_input": question})
print(result)