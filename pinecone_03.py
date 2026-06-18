import os
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
from sentence_transformers import SentenceTransformer

load_dotenv()
api_key = os.getenv("PINECONE_API_KEY")

pc = Pinecone(api_key=api_key)

index_name = "my-first-index"

if index_name not in pc.list_indexes().names():
    pc.create_index(
        name = index_name,
        dimension=384,
        metric='cosine',
        spec=ServerlessSpec(
            cloud='aws',
            region='us-east-1'
        )
    )
    print(f"Index '{index_name}' created")
else:
    print(f"Index '{index_name}' already exists")

# Connect to the index
index = pc.Index(index_name)

print(f"\nIndex stats: {index.describe_index_stats()}")

model = SentenceTransformer('all-MiniLM-L6-v2')

# Our documents with metadata
documents = [
    {"id": "doc_0", "text": "Python is a popular programming language for AI and data science", "topic": "python"},
    {"id": "doc_1", "text": "Machine learning models learn patterns from data", "topic": "ml"},
    {"id": "doc_2", "text": "Neural networks are inspired by the human brain", "topic": "ml"},
    {"id": "doc_3", "text": "LangChain helps build applications with language models", "topic": "llm"},
    {"id": "doc_4", "text": "Vector databases store embeddings for semantic search", "topic": "vectordb"},
    {"id": "doc_5", "text": "FastAPI is a modern web framework for building APIs", "topic": "backend"},
    {"id": "doc_6", "text": "Docker helps containerize applications for deployment", "topic": "devops"},
    {"id": "doc_7", "text": "RAG combines retrieval with language model generation", "topic": "rag"},
    {"id": "doc_8", "text": "Transformers revolutionized natural language processing", "topic": "llm"},
    {"id": "doc_9", "text": "Embeddings represent text as numerical vectors", "topic": "vectordb"},
]

vectors = []
for doc in documents:
    embedding = model.encode(doc["text"]).tolist()
    vectors.append({
        "id": doc["id"],
        "values": embedding,
        "metadata": {
            "text": doc["text"],
            "topic": doc["topic"]
        }
    })

# Upsert into Pinecone
index.upsert(vectors=vectors)

print(f"\nUpserted {len(vectors)} documents")
print(f"Index stats after upsert: {index.describe_index_stats()}")

query = "how do language models work"
query_embedding = model.encode(query).tolist()

# Search top 3
results = index.query(
    vector=query_embedding,
    top_k=3,
    include_metadata=True
)

print(f"\n--- Query: '{query}' ---")
for match in results['matches']:
    print(f"\nScore: {match['score']:.4f}")
    print(f"Text: {match['metadata']['text']}")
    print(f"Topic: {match['metadata']['topic']}")

# Query WITH metadata filter
results_filtered = index.query(
    vector=query_embedding,
    top_k=2,
    include_metadata=True,
    filter={"topic": {"$eq": "llm"}}
)

print(f"\n--- Query WITH filter (topic = llm only) ---")
for match in results_filtered['matches']:
    print(f"\nScore: {match['score']:.4f}")
    print(f"Text: {match['metadata']['text']}")
    print(f"Topic: {match['metadata']['topic']}")
    
user_docs = {
    "user_alice": [
        {"id": "a_0", "text": "Alice's project is about computer vision", "owner": "alice"},
        {"id": "a_1", "text": "Alice is working on image classification models", "owner": "alice"},
    ],
    "user_bob": [
        {"id": "b_0", "text": "Bob is building a chatbot with LangChain", "owner": "bob"},
        {"id": "b_1", "text": "Bob's project uses RAG for document search", "owner": "bob"},
    ]
}

# Upsert each user's docs into their own namespace
for namespace, docs in user_docs.items():
    vectors = []
    for doc in docs:
        embedding = model.encode(doc["text"]).tolist()
        vectors.append({
            "id": doc["id"],
            "values": embedding,
            "metadata": {"text": doc["text"], "owner": doc["owner"]}
        })
    index.upsert(vectors=vectors, namespace=namespace)
    print(f"Upserted {len(vectors)} docs into namespace '{namespace}'")

# Query ONLY inside Bob's namespace
query = "document search project"
query_embedding = model.encode(query).tolist()

bob_results = index.query(
    vector=query_embedding,
    top_k=2,
    include_metadata=True,
    namespace="user_bob"
)

print(f"\n--- Searching in Bob's namespace only ---")
for match in bob_results['matches']:
    print(f"Score: {match['score']:.4f} | {match['metadata']['text']}")

# Same query in Alice's namespace
alice_results = index.query(
    vector=query_embedding,
    top_k=2,
    include_metadata=True,
    namespace="user_alice"
)

print(f"\n--- Same query in Alice's namespace ---")
for match in alice_results['matches']:
    print(f"Score: {match['score']:.4f} | {match['metadata']['text']}")