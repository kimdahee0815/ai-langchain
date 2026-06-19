# 입력 1개                  복사되어 2라인 실행                          출력 dict 1개
# {"question": q} ----->     answer_chain  (prompt|model|parser)  ----> "answer": str
#                   /                                                                    ---> {"answer": ..., "faq": ...}
#                   ---->    faq_chain   (pronpt|nodel|parser)    --->   "faq":  str

from langchain_core.runnables import RunnableParallel
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.chat_models import init_chat_model
from dotenv import load_dotenv

load_dotenv()

model = init_chat_model("openai:gpt-5.4-nano")

# 창구 1: 답변 라인
answer_prompt = ChatPromptTemplate.from_messages([
    ("human", "당신은 친절한 코치입니다. 다음 질문에 3문장 이내로 답하세요: {question}")
])
answer_chain = answer_prompt | model | StrOutputParser()

# 창구 2: FAQ 라인
faq_prompt = ChatPromptTemplate.from_messages([
    ("human", "{question} - 이 질문을 한 사람이 이어서 궁금해할 만한 FAQ 질문 1개를 한 문장으로 만드세요.")
])
faq_chain = faq_prompt | model | StrOutputParser()

q = {
    "question": "저는 고객 미팅에서 이탈 징후를 먼저 발견하고, "
        "데이터 근거를 정리해 설득한 끝에 재계약을 끌어냈습니다. 이런 걸 어떻게 resume에 녹여 쓰면 좋을까요?"
}
parallel = RunnableParallel(answer= answer_chain, faq=faq_chain)

parallel_result = parallel.invoke(q)

print(parallel_result)