from dotenv import load_dotenv
import os

import bs4
from langchain import hub
from langchain.chains.llm import LLMChain
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableSequence
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import PromptTemplate

from langchain_teddynote.messages import stream_response

from langchain_community.retrievers.bm25 import BM25Retriever

from langchain.retrievers.self_query.base import SelfQueryRetriever

from NaverSearch import search_naver

load_dotenv()

from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model_name='gpt-3.5-turbo', temperature=0, max_tokens=4096).bind(logprobs=True)

search_prompt = PromptTemplate.from_template(
"""
질문을 대답하기 위해 검색해야 할 핵심 키워드를 하나만 추출해주세요. (ex. #Question: 뮤지컬 해적의 캐스팅을 알려줘 #Query: 뮤지컬 해적)
#Question: 
{question} 
#Query:
"""
)


question = "2022년 뮤지컬 라흐헤스트의 캐스팅을 불렛포인트 형식으로 배역과 함께 알려줘"
# question = input("질문을 입력하세요.\n")

search_chain = (
    {"question": RunnablePassthrough()}
    | search_prompt
    | llm
    | StrOutputParser()
)

search_query = search_chain.invoke(question)

# print(search_query)

# 뉴스기사 내용을 로드하고, 청크로 나누고, 인덱싱합니다.
loader = WebBaseLoader(
    web_paths=(search_naver(search_query)),
    bs_kwargs=dict(
        parse_only=bs4.SoupStrainer("body"),
    ),
    verify_ssl=False,
)

docs = loader.load()
# print(f"문서의 수: {len(docs)}")
# print(docs)

text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
splits = text_splitter.split_documents(docs)
print(f"분할 청크의 수: {len(splits)}")
# vectorstore = Chroma.from_documents(documents=splits, embedding=OpenAIEmbeddings())


# retriever = vectorstore.as_retriever(
#     search_kwargs={
#         "k": 10
#     }
# )

# llm_chain = LLMChain(llm=llm, prompt=prompt_template)
#
# retriever = SelfQueryRetriever(
#     vectorstore=vectorstore,
#     llm_chain=llm_chain
# )
retriever = BM25Retriever.from_documents(splits, k=5)


print(retriever)

from langchain_core.prompts import PromptTemplate

prompt = PromptTemplate.from_template(
    """당신은 질문-답변(Question-Answering)을 수행하는 친절한 AI 어시스턴트입니다. 당신의 임무는 주어진 문맥(context) 에서 주어진 질문(question) 에 답하는 것입니다.
검색된 다음 문맥(context) 을 사용하여 질문(question)에 가능한 한 세부적인 내용까지 답변하세요. 만약, 주어진 문맥(context) 에서 답을 찾을 수 없다면, 답을 모른다면 `주어진 정보에서 질문에 대한 정보를 찾을 수 없습니다` 라고 답하세요.
모든 답변은 한글로 작성해 주세요. 단, 기술적인 용어나 이름은 번역하지 않고 그대로 사용해 주세요.

#Question: 
{question} 

#Context: 
{context} 

#Answer:"""
)

# 체인을 생성합니다.
rag_chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)


answer = rag_chain.invoke(question)
print(f"Q. {question}\nA.{answer}\n")