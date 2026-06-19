from typing import Any
from typing import TypedDict
from langgraph.graph import END
from langgraph.graph import START
from langgraph.graph import StateGraph

class RagState(TypedDict):
    question: str     # 사용자의 질문(입력 값)
    docs: list[Any]   # 벡터디비에서 검색된 문서 조각들
    answer: str       # 최종 답변 (출력 값)
    quality: dict[str, Any] # 평가 결과 - 통과 여부 + 이유
    attempts: int     # 재검색 시도 횟수
    route_log: list[str] # 분기 기록
    
def retriever(state: RagState) -> dict[str, Any]:
    """질문으로 문서를 검색한다. 지금은 fake 문서를 리턴"""
    question = state["question"]
    docs = [{"text":f"fake retrieved context for : {question}"}]
    return {"docs":docs}

def grade(state: RagState) -> dict[str, Any]:
    """검색 결과를 평가한다. 지금은 문서가 있는지만 체크"""
    docs = state.get("docs", [])
    quality = {"passed": bool(docs), "reason": "compile skeleton check"}
    return {"quality": quality}

def generate(state: RagState) -> dict[str, Any]:
    """답변을 생성한다. 지금은 LLM 없이 임시 답변"""
    question = state["question"]
    return {"answer": f"임시 답변: {question}"}

builder = StateGraph(RagState)
builder.add_node("retrieve",retriever)
builder.add_node("grade", grade)
builder.add_node("generate", generate)

builder.add_edge(START, "retrieve")
builder.add_edge("retrieve","grade")
builder.add_edge("generate",END)

graph = builder.compile()
print("graph compile 통과")

result = graph.invoke({
    "question": "휴가 규정은?",
    "docs": [],
    "answer": "",
    "quality": {},
    "attempts": 0,
    "route_log": []
})

print(result["answer"])
print(result["quality"])