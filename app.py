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

    # Enhanced system prompt for more natural and comprehensive responses
    system_prompt = """
You are an intelligent and helpful AI assistant providing document-based information.

When responding to questions:
1. Address the user as "Dear user" and maintain a friendly, formal, and professional tone
2. Provide comprehensive and detailed answers based on the given context
3. Organize your responses with clear structure when appropriate
4. Speak naturally and conversationally while maintaining professionalism
5. If the information is not in the context, politely acknowledge this limitation
6. End your responses with a friendly closing remark or offer for further assistance

Your goal is to make the user feel comfortable and valued while providing accurate information.
"""

    # Generate response using Groq
    response = groq_client.chat.completions.create(
        model="llama3-70b-8192",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"""Use the following pieces of information to answer the question. If the information is not in the context, politely say you don't have enough information.

Context:
{context}

Question: {question}"""}
        ]
    )

    return response.choices[0].message.content

# App title and description
st.title("📚 Document Intelligence Assistant")
st.markdown("""
### Welcome to your Personal Document Assistant

This intelligent application helps you extract insights from your documents using advanced AI technology.

**How it works:**
The system uses Retrieval Augmented Generation (RAG) to provide accurate, context-aware answers to your questions about document content.

Simply upload your PDF document and start a conversation about its contents.
""")

# Sidebar for document upload
with st.sidebar:
    st.header("📁 Document Management")

    st.markdown("Please upload your document to begin analysis.")

    # File uploader
    uploaded_file = st.file_uploader("Select PDF Document", type="pdf")

# Main content area
col1, col2 = st.columns([3, 2])

# Document processing section
with col1:
    st.header("📊 Document Analysis")

    if uploaded_file is not None:
        # Display file info with more professional formatting
        st.success(f"Document Ready: **{uploaded_file.name}**")
        st.markdown(f"*Size: {uploaded_file.size/1024:.1f} KB | Type: PDF*")

        # Process button with more professional text
        if st.button("📝 Analyze Document Content"):
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
                    st.success(f"🎉 Analysis Complete! Successfully processed {len(chunks)} content segments from your document.")
                    st.info("💡 You can now ask questions about the document content in the consultation panel.")
                except Exception as e:
                    st.error(f"🚨 Document Analysis Issue: {str(e)}\n\nPlease try again with a different document or contact support if the problem persists.")
    else:
        st.info("📄 Please upload a PDF document to begin the analysis process.")

# Q&A section
with col2:
    st.header("💬 Document Consultation")

    # Question input with more professional prompt
    question = st.text_input("What would you like to know about this document?")

    # Answer button with more professional text
    if st.button("🔍 Generate Comprehensive Answer"):
        if question:
            if uploaded_file is not None and st.session_state.get('document_processed', False):
                with st.spinner("Generating answer..."):
                    try:
                        # Retrieve relevant chunks
                        relevant_chunks = retrieve_chunks(question, st.session_state.chunks)

                        # Generate response
                        answer = generate_response(question, relevant_chunks)

                        # Display answer with better formatting
                        st.subheader("💬 Response:")
                        st.markdown(f"""{answer}""")

                        # Display sources with better formatting
                        with st.expander("📗 View Reference Sources"):
                            st.markdown("The response was generated based on these sections from your document:")
                            for i, chunk in enumerate(relevant_chunks):
                                st.markdown(f"**Reference {i+1}:**")
                                st.markdown(f"""```
{chunk["text"][:300] + "..." if len(chunk["text"]) > 300 else chunk["text"]}
```""")
                                st.markdown(f"*Page {chunk['metadata'].get('page', 'N/A')}*")
                    except Exception as e:
                        st.error(f"🚨 Response Generation Issue: {str(e)}\n\nPlease try again or rephrase your question.")
            else:
                st.info("📝 Please upload and analyze a document before asking questions.")
        else:
            st.info("🔎 Please enter your question about the document.")

# Display usage instructions
st.markdown("""
### Getting Started:

**Step 1:** Upload your PDF document using the sidebar panel
**Step 2:** Click "Process Document" to analyze and index the content
**Step 3:** Type your question about the document in the text field
**Step 4:** Click "Get Answer" to receive a comprehensive response

*For optimal results, ask specific questions related to the document content.*
""")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p>Powered by advanced AI technology | Designed to enhance your document analysis experience</p>
    <p>© 2025 Document Intelligence Assistant</p>
</div>
""", unsafe_allow_html=True)

if __name__ == "__main__":
    # This will only run when the script is executed directly
    pass
