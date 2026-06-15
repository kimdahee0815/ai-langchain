from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.chat_models import init_chat_model

import asyncio

from dotenv import load_dotenv

load_dotenv()

prompt = ChatPromptTemplate.from_messages([
    ("system", "당신은 친절한 기술 튜터입니다."),
    ("human", "{topic}을 초보자에게 3문장으로 설명하세요.")
])

model = init_chat_model("openai:gpt-5.4-nano")

parser = StrOutputParser()

INPUT = {"topic": "LCEL"}

chain = prompt | model | parser

async def main():
    # await: 응답이 완성될 때까지 이 지점에서 "비동기적으로" 기다린다.
    return await chain.ainvoke(INPUT)

ainvoke_result = asyncio.run(main()) # 비동기 세계로 들어가는 표준 진입

print("[ainvoke]", type(ainvoke_result))
print("[ainvoke]", ainvoke_result)