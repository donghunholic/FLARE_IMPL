from dotenv import load_dotenv
import os

from langchain.memory import ConversationBufferMemory
from langchain_openai import ChatOpenAI
from langchain_teddynote.messages import stream_response
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.chains import ConversationChain


load_dotenv()

llm = ChatOpenAI(model_name='gpt-3.5-turbo', temperature=0.3, max_tokens=2048).bind(logprobs=True)
# ans = llm.invoke("트럼프 피격 사건에 대해 알려줘")
# ans = llm.stream("트럼프 피격 사건에 대해 알려줘")
# stream_response(ans)

# print(ans)
# print(ans.response_metadata)

# prompt = PromptTemplate.from_template("뮤지컬 배우 {sub}에 대해 1000자 분량으로 알려줘")
# #prompt = prompt_template.format(sub="윤하")
#
# outputparser = StrOutputParser()
#
# chain = prompt | llm | outputparser
#
# ans = chain.invoke({"sub": "이봄소리"})
# print(ans)

conv = ConversationChain(
    llm=llm,
    memory=ConversationBufferMemory()
)


response = conv.predict(
    input="안녕하세요, 비대면으로 은행 계좌를 개설하고 싶습니다. 어떻게 시작해야 하나요?"
)
print(response)

response = conv.predict(
    input="이전 답변을 불렛포인트 형식으로 정리하여 알려주세요."
)
print(response)
