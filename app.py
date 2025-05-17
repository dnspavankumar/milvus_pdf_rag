import os
import streamlit as st
import tempfile
import numpy as np
import PyPDF2
from dotenv import load_dotenv
from groq import Groq

# Load environment variables
load_dotenv()

# Set page configuration
st.set_page_config(
    page_title="Document Q&A with RAG",
    page_icon="📚",
    layout="wide"
)

# Initialize Groq client
@st.cache_resource
def initialize_client():
    return Groq(api_key=os.getenv("GROQ_API_KEY"))

groq_client = initialize_client()

# Function to generate embeddings using Groq's API
@st.cache_data(ttl=3600)
def generate_embedding(text):
    # Use the existing groq_client
    # For text embedding, we'll use the chat model to generate a summary
    # and then use that as a simple vector representation
    response = groq_client.chat.completions.create(
        model="llama3-8b-8192",  # Using a smaller model for faster processing
        messages=[
            {"role": "system", "content": "Summarize the following text in exactly 10 key terms, separated by commas. Only output the comma-separated list, nothing else."},
            {"role": "user", "content": text}
        ],
        max_tokens=100
    )

    # Get the summary as a list of terms
    summary = response.choices[0].message.content.strip()
    terms = [term.strip() for term in summary.split(',')]

    # Create a simple binary vector based on the presence of terms
    # This is a very simplified approach but should work for basic similarity
    vector = [1.0 if term in text.lower() else 0.0 for term in terms]

    # Ensure we have a vector of consistent length (pad if necessary)
    while len(vector) < 10:
        vector.append(0.0)

    return vector[:10]  # Return a fixed-length vector

# Function to calculate cosine similarity
def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# Function to process document
def process_document(file_path):
    # Extract text from PDF
    text_chunks = []

    with open(file_path, 'rb') as file:
        pdf_reader = PyPDF2.PdfReader(file)

        # Process each page
        for page_num, page in enumerate(pdf_reader.pages):
            text = page.extract_text()

            if text.strip():  # Only process non-empty pages
                # Split text into smaller chunks (paragraphs)
                paragraphs = text.split('\n\n')

                for i, para in enumerate(paragraphs):
                    if para.strip():  # Skip empty paragraphs
                        text_chunks.append({
                            "text": para.strip(),
                            "metadata": {"page": page_num + 1, "chunk": i}
                        })

    # Generate embeddings for each chunk
    chunks = []
    for chunk in text_chunks:
        chunks.append({
            "text": chunk["text"],
            "embedding": generate_embedding(chunk["text"]),
            "metadata": chunk["metadata"]
        })

    return chunks

# Function to retrieve relevant chunks
def retrieve_chunks(question, chunks, top_k=3):
    question_embedding = generate_embedding(question)

    # Calculate similarities
    similarities = []
    for i, chunk in enumerate(chunks):
        similarity = cosine_similarity(question_embedding, chunk["embedding"])
        similarities.append((i, similarity))

    # Sort by similarity (descending)
    similarities.sort(key=lambda x: x[1], reverse=True)

    # Return top k chunks
    return [chunks[i] for i, _ in similarities[:top_k]]

# Function to generate response
def generate_response(question, chunks):
    # Prepare context from chunks
    context = "\n\n".join([f"Text: {chunk['text']}" for chunk in chunks])

    # Generate response using Groq
    response = groq_client.chat.completions.create(
        model="llama3-70b-8192",
        messages=[
            {"role": "system", "content": "You are an AI assistant. Provide answers based on the given context."},
            {"role": "user", "content": f"""Use the following pieces of information to answer the question. If the information is not in the context, say you don't know.

Context:
{context}

Question: {question}"""}
        ]
    )

    return response.choices[0].message.content

# App title and description
st.title("📚 Document Q&A with RAG")
st.markdown("""
This application uses Retrieval Augmented Generation (RAG) to answer questions about your documents.
Upload a PDF file, and then ask questions about its content.
""")

# Sidebar for document upload
with st.sidebar:
    st.header("Document Management")

    # File uploader
    uploaded_file = st.file_uploader("Upload a PDF document", type="pdf")

# Main content area
col1, col2 = st.columns([3, 2])

# Document processing section
with col1:
    st.header("Document Processing")

    if uploaded_file is not None:
        # Display file info
        st.info(f"File: {uploaded_file.name} ({uploaded_file.size} bytes)")

        # Process button
        if st.button("Process Document"):
            with st.spinner("Processing document..."):
                try:
                    # Save uploaded file to a temporary file
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                        tmp_file.write(uploaded_file.getvalue())
                        pdf_path = tmp_file.name

                    # Process document
                    chunks = process_document(pdf_path)

                    # Store chunks in session state
                    st.session_state.chunks = chunks

                    # Clean up temporary file
                    os.unlink(pdf_path)

                    st.session_state.document_processed = True
                    st.success(f"Document processed successfully! Extracted {len(chunks)} chunks.")
                except Exception as e:
                    st.error(f"Error processing document: {str(e)}")
    else:
        st.warning("Please upload a PDF document.")

# Q&A section
with col2:
    st.header("Ask Questions")

    # Question input
    question = st.text_input("Enter your question about the document")

    # Answer button
    if st.button("Get Answer"):
        if question:
            if uploaded_file is not None and st.session_state.get('document_processed', False):
                with st.spinner("Generating answer..."):
                    try:
                        # Retrieve relevant chunks
                        relevant_chunks = retrieve_chunks(question, st.session_state.chunks)

                        # Generate response
                        answer = generate_response(question, relevant_chunks)

                        # Display answer
                        st.subheader("Answer:")
                        st.write(answer)

                        # Display sources (optional)
                        with st.expander("View sources"):
                            for i, chunk in enumerate(relevant_chunks):
                                st.markdown(f"**Source {i+1}:**")
                                st.text(chunk["text"][:300] + "..." if len(chunk["text"]) > 300 else chunk["text"])
                    except Exception as e:
                        st.error(f"Error generating answer: {str(e)}")
            else:
                st.warning("Please upload and process a document first.")
        else:
            st.warning("Please enter a question.")

# Display usage instructions
st.markdown("""
### How to use this app:
1. Upload a PDF document using the sidebar
2. Click "Process Document" to extract and index the content
3. Enter your question in the text box
4. Click "Get Answer" to generate a response based on the document content
""")

# Footer
st.markdown("---")
st.markdown("Built with Groq and Streamlit")

if __name__ == "__main__":
    # This will only run when the script is executed directly
    pass
