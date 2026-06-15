from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from models import get_model

model = get_model()

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "당신은 친절한 기술 튜터입니다."),
        ("human", "LCEL을 초보자에게 3문장으로 설명해 주세요.")
    ]
)

parser = StrOutputParser()

chain = prompt | model | parser
# chain = prompt | parser | model xxxx 안됨. parser는 맨 마지막
# dict -> prompt -> ChatPromptValue -> model -> AIMessage -> parser -> str
print(type(chain))