import os
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
from pinecone_text.sparse import BM25Encoder
from sentence_transformers import SentenceTransformer
load_dotenv()

api_key = os.getenv("PINECONE_API_KEY")
client = Pinecone(api_key=api_key)

index_name = "hybrid-search-index"

if index_name not in client.list_indexes().names():
    client.create_index(
        name=index_name,
        dimension=384,
        metric='dotproduct',
        spec=ServerlessSpec(
            cloud='aws',
            region = 'us-east-1'
        )
    )
    print(f"Index '{index_name}' created")
else:
    print(f"Index '{index_name}' already exists")
    
index = client.Index(index_name)

documents = [
    "Python 3.10 release notes and new features",
    "LangChain v0.2.1 changelog and bug fixes",
    "Machine learning models learn patterns from data",
    "FastAPI 0.100.0 introduced major performance improvements",
    "Neural networks are inspired by the human brain",
    "Docker 24.0 release with new container features",
    "RAG combines retrieval with language model generation",
    "LangChain helps build applications with language models",
    "Vector databases store embeddings for semantic search",
    "Python is great for data science and AI development",
]

print(f"\nFitting BM25 on documents...")

bm25 = BM25Encoder()
bm25.fit(documents)

print("BM25 fitted successfully")

# Load dense embedding model
model = SentenceTransformer('all-MiniLM-L6-v2')

vectors = []
for i, doc in enumerate(documents):
    dense_vec = model.encode(doc).tolist()
    sparse_vec = bm25.encode_documents([doc])[0]
    
    vectors.append({
        "id": f"doc_{i}",
        "values": dense_vec,
        "sparse_values": sparse_vec,
        "metadata": {"text": doc}
    })

# Upsert into Pinecone
index.upsert(vectors=vectors)
print(f"\nUpserted {len(vectors)} documents with hybrid vectors")
print(f"Index stats: {index.describe_index_stats()}")

def hybrid_query(query_text, alpha, top_k=3):
    # Get dense and sparse vectors for query
    dense_vec = model.encode(query_text).tolist()
    sparse_vec = bm25.encode_queries(query_text)
    
    # Scale vectors by alpha (this is how Pinecone does hybrid weighting)
    dense_vec = [v * alpha for v in dense_vec]
    sparse_vec["values"] = [v * (1 - alpha) for v in sparse_vec["values"]]
    
    results = index.query(
        vector=dense_vec,
        sparse_vector=sparse_vec,
        top_k=top_k,
        include_metadata=True
    )
    return results

# Test query — exact keyword match scenario
query = "LangChain v0.2.1"

print(f"\n=== Query: '{query}' ===\n")

print("--- alpha=1.0 (pure semantic search) ---")
results = hybrid_query(query, alpha=1.0)
for match in results['matches']:
    print(f"Score: {match['score']:.4f} | {match['metadata']['text']}")

print("\n--- alpha=0.0 (pure keyword search) ---")
results = hybrid_query(query, alpha=0.0)
for match in results['matches']:
    print(f"Score: {match['score']:.4f} | {match['metadata']['text']}")

print("\n--- alpha=0.5 (balanced hybrid) ---")
results = hybrid_query(query, alpha=0.5)
for match in results['matches']:
    print(f"Score: {match['score']:.4f} | {match['metadata']['text']}")