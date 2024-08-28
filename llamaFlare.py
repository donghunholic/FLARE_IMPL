from llama_index.llms.openai import OpenAI
from llama_index.core import Settings
from dotenv import load_dotenv

load_dotenv()

import logging
import sys

# logging.basicConfig(stream=sys.stdout, level=logging.INFO)
# logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

import txtai, os

# Create txtai ann index
txtai_index = txtai.ann.ANNFactory.create({"backend": "numpy"})

from llama_index.core import (
    SimpleDirectoryReader,
    load_index_from_storage,
    VectorStoreIndex,
    StorageContext,
)
from llama_index.vector_stores.txtai import TxtaiVectorStore
from IPython.display import Markdown, display

Settings.llm = OpenAI(model="gpt-3.5-turbo", temperature=0)
Settings.chunk_size = 512

from llama_index.core import SimpleDirectoryReader, VectorStoreIndex


# load documents
# documents = SimpleDirectoryReader("./data/wiki.txt").load_data()
documents = SimpleDirectoryReader("./data/paul_graham/").load_data()
# documents = SimpleDirectoryReader("./data/wiki/").load_data()
index = VectorStoreIndex.from_documents(
    documents,
)

# # 'storage' 폴더가 없으면
# if not os.path.exists("./storage"):
#     documents = SimpleDirectoryReader("./data/wiki/").load_data()
#     index = VectorStoreIndex.from_documents(documents)
#     # 나중에 사용할 수 있도록 저장
#     index.storage_context.persist()
# else:
#     # 저장된 인덱스 로드
#     storage_context = StorageContext.from_defaults(persist_dir="./storage")
#     index = load_index_from_storage(storage_context)

index_query_engine = index.as_query_engine(similarity_top_k=2)

from llama_index.core.query_engine import FLAREInstructQueryEngine

flare_query_engine = FLAREInstructQueryEngine(
    query_engine=index_query_engine,
    max_iterations=5,
    verbose=True,
)
response = flare_query_engine.query(
    "오바마는 문재인을 만난 적이 있나?"
)

print(response)

# response = flare_query_engine.query(
#     "would you please summarize the story?"
# )
#
# print(response)