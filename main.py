import argparse
from legal_docs.indexing import index_to_qdrant
from legal_docs.retrieval import QdrantRetrievalEngine
from langchain_openai import OpenAIEmbeddings

def main():
    parser = argparse.ArgumentParser(description="Legal Document Indexing and Retrieval")
    subparsers = parser.add_subparsers(dest="command")
    
    # Index command
    index_parser = subparsers.add_parser("index", help="Index a document")
    index_parser.add_argument("url", help="URL of the document to index")
    
    # Query command
    query_parser = subparsers.add_parser("query", help="Query documents")
    query_parser.add_argument("query", help="Search query")
    query_parser.add_argument("-k", type=int, default=5, help="Number of results")
    query_parser.add_argument("--source", help="Filter by source URL")
    
    args = parser.parse_args()
    
    if args.command == "index":
        print(f"Indexing document from {args.url}")
        case_docs = index_to_qdrant(args.url)
        print(f"Indexed {len(case_docs)} documents")
        
    elif args.command == "query":
        print(f"Querying: '{args.query}'")
        
        # Initialize retrieval engine
        embedding_model = OpenAIEmbeddings(model="text-embedding-3-large")
        engine = QdrantRetrievalEngine(embedding_model)
        
        # Retrieve results
        results = engine.retrieve_cases(
            args.query, 
            k=args.k,
            source_filter=args.source
        )
        
        # Display results
        print(f"\nTop {len(results)} results:")
        for i, res in enumerate(results):
            print(f"\n#{i+1} (Score: {res.score:.4f})")
            print(f"Content: {res.case.content[:200]}...")
            print(f"Source: {res.case.metadata['source']}")
            print("-" * 60)
            
    else:
        parser.print_help()

if __name__ == "__main__":
    main()