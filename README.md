# RAG with Milvus and Groq

A Retrieval Augmented Generation (RAG) application that allows you to ask questions about your PDF documents.

## Features

- Upload and process PDF documents
- Index document content using Milvus vector database
- Generate embeddings and answers using Groq AI
- Simple web interface built with Streamlit

## Installation

1. Clone this repository
2. Install the required packages:

```bash
pip install -r requirements.txt
```

3. Create a `.env` file with your Groq API key:

```
GROQ_API_KEY=your_groq_api_key_here
MILVUS_URI=./milvus_demo.db
PDF_FILE_PATH=./pdf_files/WhatisMilvus.pdf
```

## Usage

1. Run the application:

```bash
# On Windows
run_app.bat

# On macOS/Linux
chmod +x run_app.sh
./run_app.sh
```

2. Open your browser and go to http://localhost:8501
3. Upload a PDF document
4. Click "Process Document" to extract and index the content
5. Ask questions about the document and get AI-generated answers

## How It Works

This application uses:

- **Unstructured**: To extract and chunk content from PDF documents
- **Milvus**: A vector database to store and search document embeddings
- **Groq**: For generating embeddings and answering questions
- **Streamlit**: For the web interface

The RAG pipeline works as follows:
1. Document is processed and split into chunks
2. Each chunk is converted to an embedding vector using Groq
3. Embeddings are stored in Milvus
4. When a question is asked, it's converted to an embedding
5. Similar document chunks are retrieved from Milvus
6. Groq generates an answer based on the retrieved chunks

## License

MIT
#   m i l v u s _ p d f _ r a g  
 