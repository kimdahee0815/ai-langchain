from pydantic import BaseModel
from pydantic import Field

class SourceItem(BaseModel):
    source: str = Field(description="문서 출처 (파일명)")
    page: int = Field(default=0, description="페이지 번호")
    snippet: str = Field(description="원문 발췌")
    
class RagResponse(BaseModel):
    answer: str = Field(description="LLM이 생성한 답변")
    source: list[SourceItem] = Field(default_factory=list, description="근거 문서 목록 (top-k)")
    quality_passed: bool = Field(default=True, description="품질 검수 통과 여부")
    attempts: int = Field(default=0, description="재검색 시도 횟수")
    
class ChatRequest(BaseModel):
    """채팅 요청 공통 스키마"""
    message: str = Field(description="사용자 메시지")
    
class ChatResponse(BaseModel):
    """채팅 응답 공통 스키마"""
    reply: str = Field(description="AI 응답")

