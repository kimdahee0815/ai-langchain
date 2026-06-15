from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.chat_models import init_chat_model

from dotenv import load_dotenv

load_dotenv()

prompt = ChatPromptTemplate.from_messages([
    ("system", "당신은 친절한 기술 튜터입니다."),
    ("human", "{topic}을 초보자에게 3문장으로 설명하세요.")
])

model = init_chat_model("openai:gpt-5.4-nano")

parser = StrOutputParser()

# 조립
chain = prompt | model | parser

if __name__ =="__main__":
    result = chain.invoke({"topic": "LCEL"})
    print(type(result))
    print()
    print(result)
    