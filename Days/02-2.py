from langchain_core.runnables import RunnableParallel
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.chat_models import init_chat_model
from dotenv import load_dotenv

load_dotenv()

model = init_chat_model("openai:gpt-5.4-nano")

# [1단계] 단일 체인:
single_prompt = ChatPromptTemplate.from_messages(
    [("human", "당신은 친절한 면접 코치입니다. 다음 질문에 3문장 이내로 답하세요: {question}")]
)
single_chain = single_prompt | model | StrOutputParser()

# [2단계] 병렬 체인:
answer_prompt = ChatPromptTemplate.from_messages(
    [("human", "당신은 친절한 면접 코치입니다. 다음 질문에 3문장 이내로 답하세요: {question}")]
)
answer_chain = answer_prompt | model | StrOutputParser()

faq_prompt = ChatPromptTemplate.from_messages(
    [
        ("human", "{question} - 이 질문을 한 사람이 이어서 궁금해할 만한 FAQ 질문 1개를 한 문장으로 만드세요.")
    ]
)
faq_chain = faq_prompt | model | StrOutputParser()

parallel = RunnableParallel(answer=answer_chain, faq=faq_chain)

q = {"question": "자기소개서에서 프로젝트 경험은 어떻게 말해얗 하나요?"}

single_result = single_chain.invoke(q)
print(single_result)

print()
parallel_result = parallel.invoke(q)
print(parallel_result)
print("single : ",type(single_result))
print("parallel : ", type(parallel_result))
print(parallel_result.keys())