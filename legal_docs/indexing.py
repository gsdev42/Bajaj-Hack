from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from .models import CaseDocument
from .parser import parse_document
import numpy as np
from dotenv import load_dotenv
from typing import List
import os

load_dotenv()

def index_to_qdrant(url: str, collection_name: str = "legal_cases") -> List[CaseDocument]:
    """Index document to Qdrant and return CaseDocuments"""
    try:
        # Parse document
        split_docs = parse_document(url)
        print(f"📊 Created {len(split_docs)} chunks")

        # Load OpenAI embedding model (API key should be in environment)
        embedding_model = OpenAIEmbeddings(
            model="text-embedding-3-large",
            
        )

        # Connect to Qdrant (Cloud)
        vector_store = QdrantVectorStore.from_documents(
            documents=split_docs,
            embedding=embedding_model,
            url=os.getenv("QDRANT_URL"),
            collection_name=collection_name,
            api_key=os.getenv("QDRANT_API_KEY")  # Important for Qdrant Cloud
        )

        # Prepare CaseDocument objects
        embeddings = embedding_model.embed_documents([doc.page_content for doc in split_docs])
        case_docs = []

        for i, (doc, emb) in enumerate(zip(split_docs, embeddings)):
            case_doc = CaseDocument(
                id=f"{url}-{i}",
                content=doc.page_content,
                vector=np.array(emb),
                metadata=doc.metadata
            )
            case_docs.append(case_doc)

        print("✅ Indexing completed successfully!")
        return case_docs

    except Exception as e:
        print(f"❌ Indexing error: {e}")
        return []
