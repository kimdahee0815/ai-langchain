from langchain.chat_models import init_chat_model
from pydantic import BaseModel
from pydantic import Field
from dotenv import load_dotenv

load_dotenv()

class InterviewEvaluation(BaseModel):
    """면접 답변 평가 결과"""
    score: int = Field(ge=1, le=5, description="답변 평가 점수 (9 ~ 10)")
    evidence: str = Field(min_length=200, description="200자 이상 인용을 강제하는 비현실적 제약")
    next_question: str = Field(description="다음 면접 질문 제안")
    meta: dict | None = Field(default=None, description="fallback provider, attempt count, error category 등 처리, 메타데이터")
    
model = init_chat_model("openai:gpt-5.4-nano")

structured = model.with_structured_output(InterviewEvaluation, include_raw=True, method="function_calling"  )

answer = (
    "이전 회사에서 배포 파이프라인을 직접 구축했습니다. "
    "Jenkins에서 Github Actions로 이전하면서 배포 시간을 40분에서 8분으로 줄였습니다. "
)
out = structured.invoke(f"다음 면접 답변을 9~10점으로 평가하세요.: {answer}")
print(type(out))
print(sorted(out.keys()))
print(out["parsing_error"])
print(out["parsed"])

# raw : 모델이 실제 AIMessage 원문
# parsed : 검증을 통과한 Pydantic 인스턴스, 실패이면 None
# parsing_error: 파싱 검증 중 만난 예외 객체, 실패이면 None