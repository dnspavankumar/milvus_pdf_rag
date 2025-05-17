import os
from dotenv import load_dotenv
from groq import Groq

# Load environment variables
load_dotenv()

# Get API key
api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    print("Error: GROQ_API_KEY not found in environment variables")
    exit(1)

# Initialize Groq client
client = Groq(api_key=api_key)

# Test embedding
try:
    print("Testing embedding...")
    response = client.embeddings.create(
        input="Hello, world!",
        model="llama3-embedding-v1"
    )
    print(f"Embedding dimension: {len(response.data[0].embedding)}")
    print("Embedding test successful!")
except Exception as e:
    print(f"Embedding test failed: {str(e)}")

# Test chat completion
try:
    print("\nTesting chat completion...")
    response = client.chat.completions.create(
        model="llama3-70b-8192",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Say hello!"}
        ]
    )
    print(f"Response: {response.choices[0].message.content}")
    print("Chat completion test successful!")
except Exception as e:
    print(f"Chat completion test failed: {str(e)}")
