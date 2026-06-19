from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import TextLoader

import pathlib

# txt 는 encoding 명시
# LF, CRLF
# utf-8, cp949, euc-kr
txt_loader = TextLoader("docs/sample.txt", encoding="utf-8")
txt_docs = txt_loader.load()

print(f"len(txt_docs) = {len(txt_docs)}")
print(f"page_content[:200] = {txt_docs[0].page_content[:200]}")
print(f"metadata = {txt_docs[0].metadata}")