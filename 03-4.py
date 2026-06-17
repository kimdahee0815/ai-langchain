from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

pdf_docs = PyPDFLoader("docs/sample.pdf").load()
txt_docs = TextLoader("docs/sample.txt",
    encoding="utf-8").load()

test_chunks = None
for size in [50, 100, 200]:
    test_splitter = RecursiveCharacterTextSplitter(
        chunk_size=size,
        chunk_overlap=50,
        add_start_index = True
    )
    test_chunks = test_splitter.split_documents(pdf_docs)
    print(f"----------- chunk_size={size} --------------")
    print(f"chunk count: {len(test_chunks)}")
    
print(test_chunks[0])