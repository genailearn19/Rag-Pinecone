#==============================================================================================================
# This script demonstrates how to use Pinecone with OpenAI embeddings.
# It creates a Pinecone index, generates embeddings for a list of text content using OpenAI's embedding model,
#  and upserts the embeddings into the Pinecone index.
#==============================================================================================================
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

index_name = "rag-pinecone-index"

if not pc.has_index(index_name):
    pc.create_index(
        name=index_name,
        vector_type="dense",
        dimension=1536,
        metric="cosine",
        spec=ServerlessSpec(
            cloud="aws",
            region="us-east-1"
        ),
    )


content = [
    "Coffee is a popular beverage made from roasted beans.",
    "Tea contains antioxidants and improves health.",
    "Laptop computers are used for programming and work.",
    "Vector databases store embeddings for similarity search.",
    "Neural networks are used in deep learning models.",
    "Espresso is a strong form of coffee.",
    "Green tea is known for its health benefits.",
    "Machine learning models require training and inference.",
    "Pinecone is used for storing vector embeddings.",
    "Cappuccino contains coffee, milk, and foam.",
    "Transformers are used in modern AI systems.",
    "Databases store structured and unstructured data.",
    "Black tea has caffeine and strong flavor.",
    "Embeddings convert text into numerical vectors.",
    "AI models use GPUs for faster computation.",
    "Latte is made with espresso and milk.",
    "Semantic search finds meaning in text.",
    "Coffee beans are grown in tropical regions.",
    "Vector similarity uses cosine similarity.",
    "AI helps automate many tasks."
]

# Split the text using a period and remove empty spaces
ids = [str(id) for id,values in enumerate(content)]
print("Ids:", ids)
metadata = [{"chunk1": values} for id,values in enumerate(content) ]
print("Metadata:", metadata)

# ==========================================
# Generate OpenAI embeddings
# Model: text-embedding-3-small
# ==========================================
response = openai_client.embeddings.create(
    model="text-embedding-3-small",
    input=content
  )

embeddings = [item.embedding for item in response.data]

print("Embeddings:", embeddings)

# ==========================================
# Prepare Pinecone vectors
# ==========================================
vectors = [
    {
        "id": ids[i],
        "values": embeddings[i],
        "metadata": metadata[i]
    }
    for i in range(len(content))
]

# ==========================================
# Upsert vectors into Pinecone
# ==========================================
index = pc.Index(index_name)
upsert_response = index.upsert(vectors=vectors)

print("Upsert response:", upsert_response)
print("Embeddings successfully uploaded to Pinecone!")

# ==========================================
# Verify index statistics
# ==========================================
stats = index.describe_index_stats()
print("Index statistics:", stats)





