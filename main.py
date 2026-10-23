# ==========================================
# RETRIEVAL LOGIC
# ==========================================

from pinecone import Pinecone
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
query = "AI related information"

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

  
# ==========================================
# STEP 6: Collect retrieved text chunks
# ==========================================

context_chunks = []

for match in search_results.matches:
    text = match.metadata.get("chunk1", "")

    if text:
        context_chunks.append(text)

    print("-" * 60)
    print("ID:", match.id)
    print("Similarity Score:", round(match.score, 4))
    print("Text:", text)

# ==========================================
# STEP 7: Pass retrieved chunks to the LLM
# ==========================================

if not context_chunks:
    print("\nNo relevant information was retrieved.")

else:
    # Combine retrieved chunks into one context
    context = "\n\n".join(context_chunks)

    # Generate an answer using the retrieved context
    response = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a helpful AI assistant. "
                    "Answer the user's question using only "
                    "the provided context. If the context does "
                    "not contain enough information, clearly "
                    "say that the available information is insufficient. "
                    "Do not invent facts."
                )
            },
            {
                "role": "user",
                "content": f"""
                Context retrieved from Pinecone:
                {context}

                User Question:
                {query}

                Provide a clear and concise answer based on the context.
                """
            }
        ],
        temperature=0
    )

    # Extract the LLM's generated answer
    answer = response.choices[0].message.content

    print("\n" + "=" * 60)
    print("USER QUESTION:", query)
    print("=" * 60)
    print("\nGENERATED ANSWER:\n")
    print(answer)
