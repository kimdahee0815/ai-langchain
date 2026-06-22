import asyncio
import json
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi import Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from pydantic import Field

# chain factory
from chains import build_rag_chain
from chains import build_chat_chain
from chains import build_structured_chain
from chains import build_parallel_chain

from rag_graph import build_rag_graph
from rag_graph import make_initial_state

from rag_pipeline import (
    DEFAULT_PERSIST_DIR,
    build_index,
    format_sources,
    get_retriever
)

# 엔터티 클래스 선언
from schemas import ChatRequest
from schemas import ChatResponse
from schemas import RagResponse
from schemas import SourceItem

# chain holder
_chains: dict={}

## FastPAPI 백엔드가 시작시 1회 실행
@asynccontextmanager
async def lifespan(app: FastAPI):
    """서버 시작시 chain을 1회 생성"""
    _chains["rag"] = build_rag_chain()
    _chains["chat"] = build_chat_chain()
    _chains["structured"] = build_structured_chain()
    _chains["parallel"] = build_parallel_chain()
    _chains["rag_graph"] = build_rag_graph()

    # RAG index가 없으면 자동 빌드
    if not os.path.exists(DEFAULT_PERSIST_DIR):
        print("[backend] RAG index 생성 중....")
        await asyncio.to_thread(build_index)
        print("[backend] RAG index 생성 완료")
    
    print("[backend] chain 초기화 완료")
    yield
    _chains.clear()
    
app = FastAPI(title="RAG 사내 문서 QA 챗봇", description="LangChain LCEL + Chroma RAG", verion="1.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # frontend server ip address
    allow_methods=["*"],
    allow_headers=["*"]
)

@app.post("/rag", response_model=RagResponse)
async def rag_qa(req: ChatRequest):
    initial_state = make_initial_state(req.message)
    final_state = await asyncio.to_thread(_chains["rag_graph"].invoke, initial_state)
    sources = format_sources(final_state.get("docs", []))
    source_items = [SourceItem(**s) for s in sources]
    return RagResponse(
        answer=final_state.get("answer"),
        source=source_items,
        quality_passed=final_state.get("quality", {}).get("passed", False),
        attempts=final_state.get("attempts",0)
    )