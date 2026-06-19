import httpx
import streamlit as st

st.set_page_config(page_title="랭체인 + RAG 예제 - AI 사내문서 QnA", layout="wide", page_icon="🤖")
BACKEND_URL = "http://127.0.0.1:8000"

#Session State 초기화
if "messages" not in st.session_state:
    st.session_state.messages = []
if "mode" not in st.session_state:
    st.session_state.mode = "rag"
    
# 사이드바
with st.sidebar:
    st.title("설정")
    mode = st.radio("모드 선택", ["rag", "chat", "structured", "parallel"],
                    format_func=lambda x:{
                        "rag": "Rag 사내 문서 QA",
                        "chat": "기본 면접 코치",
                        "structured":"면접 답변 평가",
                        "parallel": "병렬 + 분기 응답"
                    }[x], index=0)
    st.session_state.mode = mode
    st.divider()

    if st.button("초기화", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    st.divider()

    st.caption("LangChain LCEL + Chroma RAG")

    mode_info={
        "rag": "사내 규정 문서를 기반으로 질문에 답합니다. \n\n예시 질문:\n- 휴가 신청 절차?\n- 출장비 한도는?",
        "chat": "면접 코치로서 자유 질문에 답합니다.",
        "structured": "면접 답변을 점수/강점/개선점으로 구조화 평가합니다.",
        "parallel": "병렬 응답(답변 + FAQ)를 동시에 생성."
    }    

    st.info(mode_info[mode])

# 메인 영역
st.title("AI 사내 문서 QA 챗봇")
st.caption("LangChain LCEL + Chroma RAG 종합 예제")
# 기존 메시지 표시
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if "sources" in msg and msg["sources"]:
            with st.expander(f"출처 ({len(msg['sources'])}건)", expanded=False):
                for src in msg["sources"]:
                    st.markdown(
                        f"**{src.get('source', 'unknown')}**"
                        f"(p.{src.get('page',0)})\n\n"
                        f"> {src.get('snippet','')}"
                    )
                    st.divider()
        if "metadata" in msg and msg['metadata']:
            with st.expander("상세 정보", expanded=False):
                st.json(msg["metadata"])
            
# 채팅 입력
if user_input := st.chat_input("질문을 입력하세요..."):
    # 사용자 메시지 표시
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)
    
    # AI 응답 처리
    with st.chat_message("assistant"):
        try:
            if st.session_state.mode == "rag":
                print("rag mode")
                with st.status("사내 문서 검색 중...", expanded=True) as status:
                    st.write("질문 분석 및 문서 검색...")
                    response = httpx.post(f"{BACKEND_URL}/rag", json={"message":user_input}, timeout=60.0)
                    response.raise_for_status()
                    data = response.json()
                    st.write(f"검색 완료 ({len(data.get('sources', []))}건)")
                    if data.get("attempts", 0) > 0:
                        st.write(f"재검색 {data['attempts']}회 수행")
                    status.update(label="검색 완료", state="complete", expanded=False)
                    answer = data.get("answer", "응답을 받지 못했습니다")
                    st.markdown(answer)
                    # 출처 카드
                    sources = data.get("sources", [])
                    if sources:
                        with st.expander(f"출처 ({len(sources)}건)", expanded=True):
                            for src in sources:
                                st.markdown(
                                    f"**{src.get('source', 'unknown')}**"
                                    f"(p.{src.get('page',0)})\n\n"
                                    f"> {src.get('snippet','')}"
                                )
                    # 메타데이터
                    metadata = {
                        "quality_passed": data.get("quality_passed", False),
                        "attempts":data.get("attempts", 0)
                    }
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer,
                        "sources": sources,
                        "metadata": metadata
                    })
            elif st.session_state.mode == "chat":
                st.write("chat mode")
                
            elif st.session_state.mode == "structured":
                st.write("structured mode")
            elif st.session_state.mode == "parallel":
                st.write("parallel mode")
        except httpx.ConnectError:
            st.error(
                "Backend 서버에 연결할 수 없습니다.\n\n"
                f"'uvicorn backend.app:app --port 8000'으로 서버를 먼저 시작하세요."
            )
        except httpx.HTTPStatusError as e:
            st.error(
                f"API 오류: {e.response.status_code}\n\n{e.response.text}"
            )
        except Exception as e:
            st.error(f"오류 발생: {e}")