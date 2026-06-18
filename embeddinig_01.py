import numpy as np
from sentence_transformers import SentenceTransformer

# Load the embedding model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Our sentences
sentences = [
    "I love dogs",
    "I adore puppies",
    "Dogs are my favorite animals",
    "The stock market crashed today",
    "Bitcoin prices are falling",
    "I enjoy playing football",
    "Soccer is a great sport",
    "Machine learning is fascinating",
    "AI is transforming the world",
    "I hate rainy days",
]

# Generate embeddings
embeddings = model.encode(sentences)

print(f"Each sentence is now a vector of {embeddings.shape[1]} numbers")
print(f"Total sentences embedded: {embeddings.shape[0]}")
print(f"\nFirst vector (first 5 numbers only): {embeddings[0][:5]}")

pairs = [
    ("I love dogs", "I adore puppies"),           # should be HIGH
    ("I love dogs", "Dogs are my favorite animals"), # should be HIGH
    ("I love dogs", "The stock market crashed today"), # should be LOW
    ("Machine learning is fascinating", "AI is transforming the world"), # should be HIGH
    ("I enjoy playing football", "Bitcoin prices are falling"), # should be LOW
]

def cosine_similarity(vector1, vector2):
    dot_product = np.dot(vector1,vector2)
    magnitude = np.linalg.norm(vector1) * np.linalg.norm(vector2)
    return dot_product/magnitude

for sentence1,sentence2 in pairs:
    index1 = sentences.index(sentence1)
    index2 = sentences.index(sentence2)
    score = cosine_similarity(embeddings[index1],embeddings[index2])
    print(f"{sentence1!r} vs {sentence2!r}")
    print(f"Score: {score:.4f}\n")
    