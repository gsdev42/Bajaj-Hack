from typing import List
from langchain.schema import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Temporary main function to avoid error
def main(url: str) -> str:
    # Replace this with your actual logic to extract text from the URL
    return "Sample text from the URL"

def parse_document(url: str) -> List[Document]:
    """Parse URL content into LangChain Documents"""
    text = main(url)
    
    if not text:
        raise ValueError("No text extracted from document")
    
    # Create parent document
    doc = Document(
        page_content=text,
        metadata={"source": url, "total_chars": len(text)}
    )
    
    # Split into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=400,
        length_function=len,
        separators=["\n\n", "\n", " ", ""]
    )
    
    return text_splitter.split_documents([doc])
