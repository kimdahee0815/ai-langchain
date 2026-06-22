"""rag_graph.py — LangGraph RAG 품질 루프 (Day 4 S5-S6)

교안 핵심 패턴:
- RagState(TypedDict) 6필드: question, docs, answer, quality, attempts, route_log
- node = state를 인자로 받고, 변경분 dict만 반환 (return docs / return state / return None 금지)
- router = state를 읽기만 하고 label 문자열만 반환 (state update 금지)
- compile = 배선 확인이지 품질 보증이 아님
- should_retry: Literal["retry", "generate"] — MAX_RETRIES cap으로 무한 루프 방지
- prepare_retry: attempts 증가 + route_log 4필드 누적 (새 list 반환, append 금지)
- route_log 4필드: attempts, quality_ok, decision, reason

배선 구조:
  START → retrieve → grade → [should_retry?]
                              ├─ "retry"    → prepare_retry → retrieve (최대 1회)
                              └─ "generate" → generate → END
"""

import os
from typing import Any, Literal, TypedDict

from langchain_core.documents import Document
from langgraph.graph import END, START, StateGraph

from chains import build_rag_chain
from rag_pipeline import (
    DEFAULT_PERSIST_DIR,
    build_index,
    format_sources,
    get_retriever,
)

# ─── 상수 ────────────────────────────────────────────────────────────

MAX_RETRIES = 1  # 재검색은 전체 실행에서 1회만 허용
MIN_SOURCES = 2  # 최소 근거 문서 수 (grade 기준)


# ─── State Schema ────────────────────────────────────────────────────

class RagState(TypedDict):
    question: str               # 사용자의 질문 (입력)
    docs: list[Any]             # 검색된 Document 조각들
    answer: str                 # 최종 답변 (출력)
    quality: dict[str, Any]     # 평가 결과 — 통과 여부 + 이유
    attempts: int               # 재검색 시도 횟수
    route_log: list[dict]       # 분기 기록 — 4필드 dict 목록


# ─── Node 함수들 ─────────────────────────────────────────────────────

def retrieve(state: RagState) -> dict[str, Any]:
    """질문으로 문서를 검색한다. rag_pipeline.get_retriever() 실제 연동."""
    question = state["question"]

    # index가 없으면 자동 빌드
    if not os.path.exists(DEFAULT_PERSIST_DIR):
        build_index()

    retriever = get_retriever(k=3)
    docs: list[Document] = retriever.invoke(question)
    return {"docs": docs}


def grade(state: RagState) -> dict[str, Any]:
    """검색 결과를 평가한다. 근거 문서가 MIN_SOURCES건 미만이면 품질 미달."""
    docs = state.get("docs", [])
    passed = len(docs) >= MIN_SOURCES
    reason = "sources ok" if passed else "insufficient sources"
    return {"quality": {"passed": passed, "reason": reason}}


def generate(state: RagState) -> dict[str, Any]:
    """답변을 생성한다. build_rag_chain()으로 실제 LLM 호출."""
    question = state["question"]
    docs = state.get("docs", [])

    # context 구성: 검색된 chunk들의 page_content를 줄바꿈으로 결합
    context = "\n\n".join(
        doc.page_content if isinstance(doc, Document) else str(doc)
        for doc in docs
    )

    # LLM 답변 생성
    rag_chain = build_rag_chain()
    answer = rag_chain.invoke({"context": context, "question": question})

    # route_log에 generate 진입 기록
    quality_ok = state.get("quality", {}).get("passed", False)
    new_log = {
        "attempts": state.get("attempts", 0),
        "quality_ok": quality_ok,
        "decision": "generate",
        "reason": "quality ok" if quality_ok else "attempts cap reached",
    }

    return {
        "answer": answer,
        "route_log": [*state.get("route_log", []), new_log],
    }


# ─── Router ──────────────────────────────────────────────────────────

def should_retry(state: RagState) -> Literal["retry", "generate"]:
    """router: state를 읽기만 하고 label 문자열만 반환. state update 금지.

    교안 규칙:
    - 이 함수 안에 대입문이 한 줄도 없어야 함
    - 반환 타입 Literal 힌트는 graph 시각화를 위한 계약
    """
    quality_ok = state.get("quality", {}).get("passed", False)
    if not quality_ok and state.get("attempts", 0) < MAX_RETRIES:
        return "retry"
    return "generate"


# ─── 변경 전담 Node ──────────────────────────────────────────────────

def prepare_retry(state: RagState) -> dict[str, Any]:
    """state 변경은 node의 책임: attempts 증가 + route_log 4필드 기록.

    교안 규칙:
    - append 금지 — 기존 list를 펼쳐서 새 list로 반환해야 누적 보장
    - in-place 변경(state["route_log"].append(...))은 채널 갱신으로 인정되지 않음
    """
    new_log = {
        "attempts": state.get("attempts", 0) + 1,
        "quality_ok": state.get("quality", {}).get("passed", False),
        "decision": "retry",
        "reason": "insufficient sources",
    }
    return {
        "attempts": state.get("attempts", 0) + 1,
        "route_log": [*state.get("route_log", []), new_log],
    }


# ─── Graph 빌드 ─────────────────────────────────────────────────────

def build_rag_graph():
    """StateGraph 기반 RAG 품질 루프 graph를 빌드하고 compile하여 반환.

    배선:
      START → retrieve → grade → [should_retry?]
                                  ├─ "retry"    → prepare_retry → retrieve
                                  └─ "generate" → generate → END
    """
    builder = StateGraph(RagState)

    # Node 등록
    builder.add_node("retrieve", retrieve)
    builder.add_node("grade", grade)
    builder.add_node("generate", generate)
    builder.add_node("prepare_retry", prepare_retry)

    # Edge 배선
    builder.add_edge(START, "retrieve")
    builder.add_edge("retrieve", "grade")
    builder.add_conditional_edges(
        "grade",
        should_retry,
        {"retry": "prepare_retry", "generate": "generate"},  # path_map: label → node
    )
    builder.add_edge("prepare_retry", "retrieve")  # 재검색 루프
    builder.add_edge("generate", END)               # 종료 경로 명시

    graph = builder.compile()
    print("[rag_graph] compile ok:", type(graph).__name__)
    return graph


# ─── 초기 state factory ─────────────────────────────────────────────

def make_initial_state(question: str) -> RagState:
    """질문 문자열로 초기 RagState를 생성."""
    return {
        "question": question,
        "docs": [],
        "answer": "",
        "quality": {},
        "attempts": 0,
        "route_log": [],
    }


# ─── smoke test ──────────────────────────────────────────────────────

if __name__ == "__main__":
    # router 단독 테스트 (graph 없이)
    case_a = {"quality": {"passed": True}, "attempts": 0}
    case_b = {"quality": {"passed": False}, "attempts": 0}
    case_c = {"quality": {"passed": False}, "attempts": 1}
    print("[router test]", should_retry(case_a), should_retry(case_b), should_retry(case_c))

    # graph 빌드
    graph = build_rag_graph()

    # 정상 질문 invoke (근거 풍부 → generate 직행 기대)
    r1 = graph.invoke(make_initial_state("연차 휴가는 며칠인가요?"))
    print("\n[정상 질문] answer:", r1["answer"][:100])
    for log in r1["route_log"]:
        print("[정상 질문] route_log:", log)

    # 다른 질문 invoke
    r2 = graph.invoke(make_initial_state("출장비 한도는 얼마인가요?"))
    print("\n[출장비 질문] answer:", r2["answer"][:100])
    for log in r2["route_log"]:
        print("[출장비 질문] route_log:", log)
