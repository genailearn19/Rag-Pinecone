# ==========================================
# RETRIEVAL LOGIC
# ==========================================

from pinecone import Pinecone
from pinecone import ServerlessSpec
from dotenv import load_dotenv
import os
from openai import OpenAI

#==========================================
# Load environment variables from .env file
#==========================================
load_dotenv()

#================================================================
# Initialize Pinecone client and create index if it doesn't exist
#================================================================
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

pc = Pinecone(api_key=PINECONE_API_KEY)
openai_client = OpenAI(api_key=OPENAI_API_KEY)

# Step 1: Define the user's query
query = "About Coffee"

index_name = "rag-pinecone-index"

# Step 2: Convert the query into an embedding
query_response = openai_client.embeddings.create(
    model="text-embedding-3-small",
    input=query
)
# Step 3:Retrieve the embedding vector from the response
query_embedding = query_response.data[0].embedding

print("Query embedding generated successfully!")

# Step 4: Connect to the existing Pinecone index
index = pc.Index(index_name)

# Step 5: Search for the most similar vectors
search_results = index.query(
    vector=query_embedding,
    top_k=3,
    include_metadata=True
)

# Step 6: Display the retrieved text chunks
print("\nQuery:", query)
print("\nRetrieved Results:")

for match in search_results.matches:
    print("-" * 60)
    print("ID:", match.id)
    print("Similarity Score:", round(match.score, 4))
    print("Text:", match.metadata.get("chunk1", "Text not found"))