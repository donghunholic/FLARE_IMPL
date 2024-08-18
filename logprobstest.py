# from openai import OpenAI
# from math import exp
# import numpy as np
# from IPython.display import display, HTML
# import os
# from dotenv import load_dotenv
#
# load_dotenv()
#
# client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY", "<your OpenAI API key if not set as env var>"))
#
# def get_completion(
#     messages: list[dict[str, str]],
#     model: str = "gpt-4",
#     max_tokens=500,
#     temperature=0,
#     stop=None,
#     seed=123,
#     tools=None,
#     logprobs=None,  # whether to return log probabilities of the output tokens or not. If true, returns the log probabilities of each output token returned in the content of message..
#     top_logprobs=None,
# ) -> str:
#     params = {
#         "model": model,
#         "messages": messages,
#         "max_tokens": max_tokens,
#         "temperature": temperature,
#         "stop": stop,
#         "seed": seed,
#         "logprobs": logprobs,
#         "top_logprobs": top_logprobs,
#     }
#     if tools:
#         params["tools"] = tools
#
#     completion = client.chat.completions.create(**params)
#     return completion
#
# CLASSIFICATION_PROMPT = """You will be given a headline of a news article.
# Classify the article into one of the following categories: Technology, Politics, Sports, and Art.
# Return only the name of the category, and nothing else.
# MAKE SURE your output is one of the four categories stated.
# Article headline: {headline}"""
#
# headlines = [
#     "Tech Giant Unveils Latest Smartphone Model with Advanced Photo-Editing Features.",
#     "Local Mayor Launches Initiative to Enhance Urban Public Transport.",
#     "Tennis Champion Showcases Hidden Talents in Symphony Orchestra Debut",
# ]
#
# for headline in headlines:
#     print(f"\nHeadline: {headline}")
#     API_RESPONSE = get_completion(
#         [{"role": "user", "content": CLASSIFICATION_PROMPT.format(headline=headline)}],
#         model="gpt-4",
#     )
#     print(f"Category: {API_RESPONSE.choices[0].message.content}\n")



import os

from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

import re

import numpy as np

from langchain.chat_models import ChatOpenAI
from langchain.llms import OpenAI
from langchain.schema import Document
from langchain_community.retrievers import BM25Retriever
import bs4

from NaverSearch import search_naver

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

# We set this so we can see what exactly is going on
import langchain

langchain.verbose = True

from langchain.chains import FlareChain

flare = FlareChain.from_llm(
    ChatOpenAI(temperature=0),
    retriever=retriever,
    max_generation_len=164,
    min_prob=0.3,
)

query = "뮤지컬 알라딘에 대해 알려줘"

flare.run(query)