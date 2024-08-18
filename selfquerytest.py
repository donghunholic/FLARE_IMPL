from dotenv import load_dotenv
import os
import bs4
from langchain.chains.flare.base import QuestionGeneratorChain
from langchain_community.retrievers import BM25Retriever
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableMap
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain.chains import FlareChain
from NaverSearch import search_naver

load_dotenv()

# OpenAI API 키 설정
os.environ['OPENAI_API_KEY'] = os.getenv('OPENAI_API_KEY')

# LLM 초기화
# llm = ChatOpenAI(model_name='gpt-3.5-turbo', temperature=0.3, max_tokens=4096, logprobs=True)

# 뉴스기사 내용을 로드하고, 청크로 나누고, 인덱싱합니다.
loader = WebBaseLoader(
    web_paths=search_naver("뮤지컬 알라딘"),
    bs_kwargs=dict(
        parse_only=bs4.SoupStrainer("body")
    ),
)

docs = loader.load()
# print(f"문서의 수: {len(docs)}")
# print(docs)

text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
splits = text_splitter.split_documents(docs)
# vectorstore = Chroma.from_documents(documents=splits, embedding=OpenAIEmbeddings())
#
# retriever = vectorstore.as_retriever(
#     search_kwargs={
#         "k": 5
#     }
# )

retriever = BM25Retriever.from_documents(splits, k=5)
print(retriever)

prompt = PromptTemplate.from_template(
    """당신은 질문-답변(Question-Answering)을 수행하는 친절한 AI 어시스턴트입니다. 당신의 임무는 주어진 문맥(context) 에서 주어진 질문(question) 에 답하는 것입니다.
검색된 다음 문맥(context) 을 사용하여 질문(question) 에 답하세요. 만약, 주어진 문맥(context) 에서 답을 찾을 수 없다면, 답을 모른다면 `주어진 정보에서 질문에 대한 정보를 찾을 수 없습니다` 라고 답하세요.
한글로 답변해 주세요. 단, 기술적인 용어나 이름은 번역하지 않고 그대로 사용해 주세요.

# Question:
{question}

# Context:
{context}

# Answer:"""
)

llm=ChatOpenAI(model_name='gpt-3.5-turbo', temperature=0.3, max_tokens=4096)
logprobs_llm = llm.bind(logprobs=True)

qa_chain = QuestionGeneratorChain(llm=llm)


flare_chain = FlareChain.from_llm(
    llm=llm,  # 언어 모델
    max_generation_len=32,  # 생성할 응답의 최대 길이
    retriever=retriever,  # 리트리버 객체
    min_prob=0.2,  # 낮은 확률 임계값
    min_token_gap=5,  # 토큰 간 최소 간격
    num_pad_tokens=2,  # 패딩 토큰 수
    max_iter=10,  # 최대 반복 횟수
    start_with_retrieval=True  # 리트리버를 먼저 사용할지 여부
)

flare_chain.run("뮤지컬 알라딘의 배역과 캐스팅을 알려줘")
