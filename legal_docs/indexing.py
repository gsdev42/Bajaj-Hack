from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from .models import CaseDocument
from .parser import parse_document
import numpy as np
from dotenv import load_dotenv
from typing import List

load_dotenv()

def index_to_qdrant(url: str, collection_name: str = "legal_cases") -> List[CaseDocument]:
    """Index document to Qdrant and return CaseDocuments"""
    try:
        # Parse document
        split_docs = parse_document(url)
        print(f"📊 Created {len(split_docs)} chunks")
        
        # Generate embeddings
        embedding_model = OpenAIEmbeddings(model="text-embedding-3-large")
        
        # Store in Qdrant
        vector_store = QdrantVectorStore.from_documents(
            documents=split_docs,
            url="http://localhost:6333",
            collection_name=collection_name,
            embedding=embedding_model
        )
        
        # Create CaseDocuments with embeddings
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