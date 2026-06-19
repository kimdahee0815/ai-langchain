from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

EMBEDDING_MODEL = "text-embedding-3-small"

pdf_docs = PyPDFLoader("docs/sample.pdf").load()
txt_docs = TextLoader("docs/sample.txt",
    encoding="utf-8").load()

splitter = RecursiveCharacterTextSplitter(
        chunk_size=200,   # 한 청크의 목표 크기 - 문자 수 기준, 토큰이 아니다.
        chunk_overlap=50,   # 인접 청크와 서로 겹치는 문자 수
        add_start_index=True  # 기본값 False - 명시해야 metadata에 start_index가 생긴다
    )

chunks = splitter.split_documents(pdf_docs)
print(f"chunks count: {len(chunks)}")

embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)

doc_vectors = embeddings.embed_documents([c.page_content for c in chunks[:5]])

print(len(doc_vectors), len(doc_vectors[0])) # 문서 수, 벡터 차원 출력