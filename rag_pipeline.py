import os
from langchain_chroma import Chroma
from langchain_community.document_loaders import TextLoader
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from models import get_model

# 상수
EMBEDDING_MODEL = "text-embedding-3-small"
DEFAULT_DOC_PATH = os.path.join(os.path.dirname(__file__), "sample_docs", "company_policy.txt")
DEFAULT_PERSIST_DIR = os.path.join(os.path.dirname(__file__),"chroma_company_docs")

# 기준 설정
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
ADD_START_INDEX = True

def get_embeddings():
    return OpenAIEmbeddings(model=EMBEDDING_MODEL)

# 인덱스 생성 : 처음 1번만 실행
def build_index(doc_path: str = DEFAULT_DOC_PATH, persist_dir: str = DEFAULT_PERSIST_DIR) -> Chroma:
    """Load -> Split -> Embed -> Chroma 저장"""
    # 1. Load
    loader = TextLoader(doc_path, encoding="utf-8")
    docs = loader.load()
    print(f"[rag_pipeline] 문서 로드: {len(docs)}")
    
    # 2. Split
    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP, add_start_index=ADD_START_INDEX)
    chunks = splitter.split_documents(docs)
    print(f"[rag_pipline] chunk 분할: {len(chunks)}건")
    
    # 3. Embed + Store
    embeddings = get_embeddings()
    db = Chroma.from_documents(chunks, embedding=embeddings, persist_directory=persist_dir)
    print(f"[rag_pipeline] index 생성 완료: {persist_dir}")
    return db

def get_retriever(persist_dir: str = DEFAULT_PERSIST_DIR, k: int = 3):
    db = Chroma(persist_directory=persist_dir, embedding_function=get_embeddings())
    return db.as_retriever(search_kwargs={"k":k})

def format_sources(docs: list[Document])->list[dict]:
    """검색된 Dcoument 리스트에서 출처 정보를 추출 SourceItem 형식으로 반환"""
    sources = []
    for doc in docs:
        sources.append({
            "source": doc.metadata.get("source", "unknown"),
            "page": doc.metadata.get("page",0),
            "snippet": doc.page_content[:50]
        })
    return sources
