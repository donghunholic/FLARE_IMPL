from dotenv import load_dotenv
import os
import bs4
from langchain.chains.llm import LLMChain
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings, OpenAI
from langchain_core.prompts import PromptTemplate

from langchain_community.retrievers.bm25 import BM25Retriever
from NaverSearch import search_naver

import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

import sys
import io

# 표준 출력 스트림의 인코딩을 UTF-8로 설정
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
print(sys.stdout.encoding)

load_dotenv()

question = "뮤지컬배우 최지혜의 작품활동에 대해 알려줘"

llm = ChatOpenAI(model_name='gpt-3.5-turbo', temperature=0, max_tokens=4096)

search_prompt = PromptTemplate.from_template(
    """
    질문을 대답하기 위해 검색해야할 핵심 키워드를 하나만 추출해주세요. (ex. #Question: 뮤지컬 해적의 캐스팅을 알려줘 #Query: 뮤지컬 해적)
    #Question:
    {question}
    #Query:
    """
)

search_chain = (
    {"question": RunnablePassthrough()}
    | search_prompt
    | llm
    | StrOutputParser()
)

search_query = search_chain.invoke(question)

loader = WebBaseLoader(
    web_paths=search_naver(search_query),
    bs_kwargs=dict(
        parse_only=bs4.SoupStrainer("body"),
    ),
    verify_ssl=False,
)

docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
splits = text_splitter.split_documents(docs)
print(f"분할 청크의 수: {len(splits)}")

retriever = BM25Retriever.from_documents(splits, k=3)

prompt = PromptTemplate.from_template(
    """당신은 질문-답변(Question-Answering)을 수행하는 친절한 AI 어시스턴트입니다. 당신의 임무는 주어진 문맥(context) 에서 주어진 질문(question)에 가능한 한 세부적인 내용까지 답변하세요. 만약, 주어진 문맥(context) 에서 답을 찾을 수 없다면, 답을 모른다면 `주어진 정보에서 질문에 대한 정보를 찾을 수 없습니다` 라고 답하세요. 모든 답변은 한글로 작성해 주세요. 단, 기술적인 용어나 이름은 번역하지 않고 그대로 사용해 주세요.

#Question: 
{question} 

#Context: 
{context} 

#Answer:"""
)

import langchain

langchain.verbose = True
from langchain.chains.flare.base import FlareChain, QuestionGeneratorChain, _OpenAIResponseChain

flare_llm = OpenAI(model_name="davinci-002", max_tokens=256, logprobs=1)
QAPromptTemplate = PromptTemplate(
    input_variables=['current_response', 'uncertain_span', 'user_input'],
    template='사용자 입력과 기존의 부분적인 응답을 맥락으로 주어졌을 때, 주어진 용어/개체/구문에 대한 답이 될 가장 중요한 질문 하나를 작성하세요:\n\n>>> USER INPUT: {user_input}\n>>> EXISTING PARTIAL RESPONSE: {current_response}\n\n"{uncertain_span}"에 대한 답이 될 질문은:'
)

question_generator_chain = QuestionGeneratorChain(llm=flare_llm, prompt=QAPromptTemplate)
response_llm = OpenAI(model_name="davinci-002", max_tokens=256, logprobs=1)
response_chain = _OpenAIResponseChain(llm=response_llm)

flare = FlareChain(
    question_generator_chain=question_generator_chain,
    response_chain=response_chain,
    retriever=retriever,
    min_prob=0.4,
    min_token_gap=5,
    num_pad_tokens=2,
    max_iter=10,
    start_with_retrieval=True
)

result = flare.invoke({"user_input": question})
print(result)