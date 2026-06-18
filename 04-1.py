from langchain_core.runnables import RunnableLambda
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
from langchain_chroma import Chroma

load_dotenv()

EMBEDDING_MODEL = "text-embedding-3-small"
embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)


def expand_queries(question: str) -> list[str]:
    """원 질문 의미를 보존한 검색 표현 2 ~ 3개를 반환"""
    return [
        question,               # [0] 원 질문은 항상 첫 자리에 보존
        f"{question} 관련 규정", # [1] 같은 의도, 표현만 변경
        f"{question} 절차 안내"  # [2] 같은 의도, 표현만 변경
    ]
    

def load_db():
    PERSIST_DIR = "./chroma_company_docs"
    db = Chroma(
        embedding_function=embeddings,
        persist_directory=PERSIST_DIR
    )
    return db

def get_retriever(k: int =3 ):
    """Chroma index를 RAG chain에 끼울 수 있는 retriever로 전환"""
    return load_db().as_retriever(search_kwargs={"k":k})
    
for i, q in enumerate(expand_queries("연차 휴가는 며칠까지 쓸 수 있나요?")):
    print(f"[{i}] {q}")

def dedup_key(doc)-> tuple:
    """같은 chunk 판별 기준 : (source, page, start_index)"""
    m = doc.metadata
    return (m.get("source"), m.get("page"), m.get("start_index"))

    
retriever = get_retriever(k=3)
    
def retrieve_for_query(query: str, k: int =3):
    """검색 호출 분리"""
    return retriever.invoke(query)

QUESTIONS = [
    "연차 휴가는 며칠까지 쓸 수 있나요?",
    "경조사 휴가 신청 절차는 어떻게 되나요?"
]

# -- 1단계: baseline - 원 질문 그대로 top3
for question in QUESTIONS:
    print(f"\n=== baseline {question} ===")
    for rank, doc in enumerate(retrieve_for_query(question), start=1):
        m = doc.metadata
        snippet = doc.page_content[:38].replace("\n", " ")
        print(f"  {rank}, {m.get('source')}, {m.get('page')}, idx={m.get('start_index')} | {snippet}")
# -- 2단계: expanded - variant 별 검색
# Langchain 파이프라인 체인에서 사용할 수 있는. Langchain tracing 자동 통합
expand_node = RunnableLambda(expand_queries)
rows = []
for question in QUESTIONS:
    seen:set[tuple] = set()
    for variant_no, query in enumerate(expand_node.invoke(question)):
        # print(variant_no, ". ",query)
        for rank, doc in enumerate(retrieve_for_query(query), start=1):
            key = dedup_key(doc)
            note = "중복" if key in seen else ""
            seen.add(key)
            snippet = doc.page_content[:38].replace("\n"," ")
            rows.append((question[:14] + "...", query[:20] + "...", variant_no, rank, *key, snippet, note))

print("\n| question | query | variant_no | rank | source | page | start_index | snippet | relevant_note")
print("|---|---|---|---|---|---|---|---|---|")
for r in rows:
    print("| "+" | ".join(str(c) for c in r) + " |")
        
queries = expand_node.invoke("연차 휴가는 며칠까지 쓸수 있나요?")
print(type(queries).__name__)
print(len(queries), "개 쿼리 준비")
