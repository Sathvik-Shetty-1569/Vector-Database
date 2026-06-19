import chromadb
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('all-MiniLM-L6-v2')
client = chromadb.PersistentClient(path="./trial")
collection = client.get_or_create_collection(name='Table')

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

metadatas = metadatas = [
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
    ids=[f"doc_{i}" for i in range(len(documents))],
    embeddings=embeddings,
    documents=documents,
    metadatas=metadatas
)
print(f"Total documents stored: {collection.count()}")

query = "how to deploy applications"

query_embeddings = model.encode(query).tolist()

result = collection.query(
    query_embeddings=query_embeddings,
    n_results=3
)
