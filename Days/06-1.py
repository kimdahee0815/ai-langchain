from typing import Any, Literal, TypedDict
from langgraph.graph import END
from langgraph.graph import START
from langgraph.graph import StateGraph

MAX_RETRIES = 1 # attempts cap: 재검색은 전체 실행에서 1회만 허용

class RagState(TypedDict):
    question: str     # 사용자의 질문(입력 값)
    docs: list[Any]   # 벡터디비에서 검색된 문서 조각들
    answer: str       # 최종 답변 (출력 값)
    quality: dict[str, Any] # 평가 결과 - 통과 여부 + 이유
    attempts: int     # 재검색 시도 횟수
    route_log: list[str] # 분기 기록
    

def should_retry(state: RagState) -> Literal["retry", "generate"]:
    """route: state를 읽기만 하고 label 문자열만 반환"""
    quality_ok = state.get("quality", {}).get("passed", False)
    print("===== Should_retry =====")
    if not quality_ok and state.get("attempts", 0) < MAX_RETRIES:
        return "retry"
    return "generate"

case_a = {"quality": {"passed": True}, "attempts": 0} # 품질 통화
case_b = {"quality": {"passed": False}, "attempts": 0} # 품질 통화 + 첫 시도
case_c = {"quality": {"passed": False}, "attempts": 1} # 품질 통화 + cap 도달

# print(should_retry(case_a))
# print(should_retry(case_b))
# print(should_retry(case_c))

def prepare_retry(state: RagState) -> dict[str, Any]:
    """state 변경은 node의 책임: attempts 증가 + route_log에 4필드 기록 추가"""
    new_log = {
        "attempts": state.get("attempts",0)+1,
        "quality_ok": state.get("quality", {}).get("passed", False),
        "decision": "retry",
        "reason": "insufficient sources"
        }
    print("===== Prepare_retry =====")
    return {
        "attempts": state.get("attempts",0)+1,
        "rotate_log": [*state.get("route_log", []), new_log]
    }
    

def grade(state: RagState) -> dict[str, Any]:
    """검색 결과를 평가한다. 지금은 문서가 있는지만 체크"""
    passed = len(state.get("docs" , [])) >=2
    reason = "sources ok" if passed else "insufficient sources"
    # quality = {"passed": bool(docs), "reason": "compile skeleton check"}
    return {"quality": {"passed": passed,"reason": reason}}

def generate(state: RagState) -> dict[str, Any]:
    """답변 생성 + 'generate'로 들어온 이유를 route_log에 남겨서 직행/우회한건지를 구분"""
    quality_ok = state.get("quality",{}).get("passed", False)
    new_log = {
        "attempts": state.get("attempts",0),
        "quality_ok": quality_ok,
        "decision": "generate",
        "reason": "quality ok" if quality_ok else "attempts cap reached"
        }
    return {
        "answer": f"임시 답변: {state['question']} (근거 {len(state.get('docs', []))}건)",
        "route_log": [*state.get("route_log", []), new_log]
    }
    
def retrieve(state: RagState) -> dict[str, Any]:
    """실제 벡터 DB 연결 = retriever.invoke(state['question'])"""
    if "연차" in state["question"]:
        docs = [
            {"text": "연차는 입사 1년 차에 15일이 부여됩니다."},
            {"text": "미사용 연차는 최대 5일까지 이월할 수 있습니다."}
        ]
    else:
        docs = [{"text": f"부분 일치 자료 1건: {state['question']}"}]
        
    return {"docs":docs}

builder = StateGraph(RagState)
builder.add_node("retrieve", retrieve)
builder.add_node("grade", grade)
builder.add_node("generate", generate)
builder.add_node("prepare_retry", prepare_retry)

builder.add_edge(START, "retrieve")
builder.add_edge("retrieve", "grade")
builder.add_conditional_edges(
    "grade",
    should_retry,
    {"retry": "prepare_retry", "generate":"generate"}
)
builder.add_edge("prepare_retry", "retrieve")
builder.add_edge("generate", END)

graph = builder.compile()
print("compile ok:", type(graph).__name__)

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