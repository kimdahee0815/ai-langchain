# closed book: 입력 = {question}
#              체인 = prompt | model | parser

# open-book:   문서조각 = 검색(question) <- 답변 전에 교재 펼치기
#              입력 = {context: 문서조각, question}
#              체인 = prompt | model | parser <- 체인은 완전히 동일

# 인덱싱 (미리, 한 번)
# Load (적재) -> Split (분할) -> Embed (임베딩) -> Store(저장)

# 런타임 (질문 순간, 매번)
# Retrieve (검색) -> Generate (생성)

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import TextLoader
import pathlib

# Document는 page_content (본문 200) + metadata(출처 꼬리표) 2개 항목으로 구성.
pdf_loader = PyPDFLoader("docs/sample.pdf") # 준비만 함 (실제 읽지는 x)
pdf_docs = pdf_loader.load() # load() 호출 시점에 실제 읽는다.

first = pdf_docs[0]
print(type(first))
print(f"필드 = page_content({type(first.page_content).__name__}), metadata({type(first.metadata).__name__})")

print(f"len(pdf_docs) = {len(pdf_docs)}")
print(f"page_content[:200] = {pdf_docs[0].page_content[:200]}")
print(f"metadata = {pdf_docs[0].metadata}")