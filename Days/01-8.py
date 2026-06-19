
from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from dotenv import load_dotenv
from pydantic import BaseModel, Field

load_dotenv()

# 모델 정의시 prefix(provider) 형식 고정
model = init_chat_model("openai:gpt-5.4-nano")

# dict 형식으로 입력을 받는다.
prompt = ChatPromptTemplate.from_messages([("human", "{q}")]) # {"q": "안녕하세요"}

chain = prompt | model | StrOutputParser() #자유 양식 출력

# 구조가 있는 출력 적용 => Pydantic
# structured_chain = prompt | model.with_structured_output(InterviewScore)

class InterviewScore(BaseModel):
    """면접 답변 평가 결과"""
    score: int = Field(ge=1, le=5, description="답변 평가 점수 (1 - 5)")
    strengths: str = Field(description="답변의 감정")
    improvements: str = Field(description="개선이 필요한 부분")
    next_question: str = Field(description="다음 면접 질문 제안")

prompt = ChatPromptTemplate.from_messages([
    ("system", "당신은 면접관입니다. 답변을 평가 기준에 따라 채점하세요."),
    ("human", "면접 답변: {answer}")
])

# 출력 계약을 바꾸는 한 줄
structured_model = model.with_structured_output(InterviewScore)

# paraser 를 뒤에 두지 않는다
chain = prompt | structured_model

if __name__ == "__main__":
    response = chain.invoke({
        "answer": "저는 고객 미팅에서 이탈 징후를 먼저 발견하고, "
        "데이터 근거를 정리해 설득한 끝에 재계약을 끌어냈습니다."
    })
    print(type(response))
    print(response.model_dump())
    result = response.model_dump()
    print(type(result))
    if result.get("score") <= 3:
        print("다시 시도하세요.")