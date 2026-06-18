import chromadb
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('all-MiniLM-L6-v2')

client = chromadb.PersistentClient(path="./chroma_db")

collection = client.get_or_create_collection(name="my_documents")

documents = [
    "Python is a popular programming language for AI and data science",
    "Machine learning models learn patterns from data",
    "Neural networks are inspired by the human brain",
    "LangChain helps build applications with language models",
    "Vector databases store embeddings for semantic search",
    "FastAPI is a modern web framework for building APIs",
    "Docker helps containerize applications for deployment",
    "RAG combines retrieval with language model generation",
    "Transformers revolutionized natural language processing",
    "Embeddings represent text as numerical vectors",
]

metadatas = [
    {"topic": "python"},
    {"topic": "ml"},
    {"topic": "ml"},
    {"topic": "llm"},
    {"topic": "vectordb"},
    {"topic": "backend"},
    {"topic": "devops"},
    {"topic": "rag"},
    {"topic": "llm"},
    {"topic": "vectordb"},
]

embeddings = model.encode(documents).tolist()

collection.upsert(
    ids=[f"doc_{i}" for i in range (len(documents))],
    embeddings=embeddings,
    documents=documents,
    metadatas= metadatas
)

print(f"Total documents stored: {collection.count()}")

query = "how to deploy applications"

query_embeddings = model.encode(query).tolist()

result = collection.query(
    query_embeddings=[query_embeddings],
    n_results=3
)

results = collection.query(
    query_embeddings=[query_embeddings],
    n_results=3
)

print(f"\n--- Query WITHOUT filter ---")
for i in range(len(results['documents'][0])):
    print(f"Rank {i+1}: {results['documents'][0][i]}")
    print(f"Topic: {results['metadatas'][0][i]['topic']} | Score: {results['distances'][0][i]:.4f}")

# Query WITH metadata filter — only search inside "llm" topic
results_filtered = collection.query(
    query_embeddings=[query_embeddings],
    n_results=2,
    where={"topic": "devops"}
)

print(f"\n--- Query WITH filter (topic = llm only) ---")
for i in range(len(results_filtered['documents'][0])):
    print(f"Rank {i+1}: {results_filtered['documents'][0][i]}")
    print(f"Topic: {results_filtered['metadatas'][0][i]['topic']} | Score: {results_filtered['distances'][0][i]:.4f}")

