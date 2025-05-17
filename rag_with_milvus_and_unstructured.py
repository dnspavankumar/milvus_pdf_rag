#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Build a RAG with Milvus and Unstructured

This script demonstrates how to build a Retrieval Augmented Generation (RAG) pipeline
using Milvus for vector storage and Unstructured for document processing.

Original source: https://github.com/milvus-io/bootcamp/blob/master/bootcamp/tutorials/integration/rag_with_milvus_and_unstructured.ipynb
"""

# Import required libraries

# Required packages:
# pip install "unstructured[pdf]" pymilvus groq

# Note: For processing all document formats: pip install "unstructured[all-docs]"
# For more installation options, see: https://docs.unstructured.io/open-source/installation/full-installation

import os
import warnings
from dotenv import load_dotenv
from pymilvus import MilvusClient, DataType
from groq import Groq
from unstructured.partition.auto import partition

# Load environment variables from .env file
load_dotenv()

# API keys and configuration are now loaded from .env file

def emb_text(text, client):
    """Generate embeddings for text using Groq.

    Args:
        text (str): The text to embed
        client (Groq): Groq client instance

    Returns:
        list: The embedding vector
    """
    return (
        client.embeddings.create(input=text, model="llama3-embedding-v1")
        .data[0]
        .embedding
    )

def retrieve_documents(question, milvus_client, collection_name, groq_client, top_k=3):
    """Retrieve relevant documents from Milvus.

    Args:
        question (str): The query question
        milvus_client (MilvusClient): Milvus client instance
        collection_name (str): Name of the collection to search
        groq_client (Groq): Groq client instance
        top_k (int): Number of results to return

    Returns:
        list: List of tuples containing (text, distance)
    """
    search_res = milvus_client.search(
        collection_name=collection_name,
        data=[emb_text(question, groq_client)],
        limit=top_k,
        output_fields=["text"],
    )
    return [(res["entity"]["text"], res["distance"]) for res in search_res[0]]


def generate_rag_response(question, milvus_client, collection_name, groq_client):
    """Generate a response using the RAG pipeline.

    Args:
        question (str): The query question
        milvus_client (MilvusClient): Milvus client instance
        collection_name (str): Name of the collection to search
        groq_client (Groq): Groq client instance

    Returns:
        str: Generated response
    """
    retrieved_docs = retrieve_documents(question, milvus_client, collection_name, groq_client)
    context = "\n".join([f"Text: {doc[0]}\n" for doc in retrieved_docs])
    system_prompt = (
        "You are an AI assistant. Provide answers based on the given context."
    )
    user_prompt = f"""
    Use the following pieces of information to answer the question. If the information is not in the context, say you don't know.

    Context:
    {context}

    Question: {question}
    """
    response = groq_client.chat.completions.create(
        model="llama3-70b-8192",  # Using Llama 3 70B model from Groq
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )
    return response.choices[0].message.content

def setup_milvus_collection(milvus_client, collection_name, embedding_dim):
    """Set up a Milvus collection for storing document embeddings.

    Args:
        milvus_client (MilvusClient): Milvus client instance
        collection_name (str): Name of the collection to create
        embedding_dim (int): Dimension of the embedding vectors

    Returns:
        None
    """
    # Check if collection exists and drop it if it does
    if milvus_client.has_collection(collection_name):
        milvus_client.drop_collection(collection_name)

    # Create schema
    schema = milvus_client.create_schema(auto_id=False, enable_dynamic_field=False)

    # Add fields to schema
    schema.add_field(field_name="id", datatype=DataType.INT64, is_primary=True)
    schema.add_field(field_name="vector", datatype=DataType.FLOAT_VECTOR, dim=embedding_dim)
    schema.add_field(field_name="text", datatype=DataType.VARCHAR, max_length=65535)
    schema.add_field(field_name="metadata", datatype=DataType.JSON)

    # Prepare index parameters
    index_params = MilvusClient.prepare_index_params()
    index_params.add_index(
        field_name="vector",
        metric_type="COSINE",
        index_type="AUTOINDEX",
    )

    # Create and load the collection
    milvus_client.create_collection(
        collection_name=collection_name,
        schema=schema,
        index_params=index_params,
        consistency_level="Strong",
    )

    milvus_client.load_collection(collection_name=collection_name)


def process_pdf_and_load_to_milvus(pdf_path, milvus_client, collection_name, groq_client):
    """Process a PDF file and load its content into Milvus.

    Args:
        pdf_path (str): Path to the PDF file
        milvus_client (MilvusClient): Milvus client instance
        collection_name (str): Name of the collection to insert data into
        groq_client (Groq): Groq client instance

    Returns:
        None
    """
    # Suppress warnings
    warnings.filterwarnings("ignore")

    # Partition the PDF file
    elements = partition(
        filename=pdf_path,
        strategy="hi_res",
        chunking_strategy="by_title",
    )

    # Prepare data for insertion
    data = []
    for i, element in enumerate(elements):
        data.append(
            {
                "id": i,
                "vector": emb_text(element.text, groq_client),
                "text": element.text,
                "metadata": element.metadata.to_dict(),
            }
        )

    # Insert data into Milvus
    milvus_client.insert(collection_name=collection_name, data=data)

def main():
    """Main function to run the RAG pipeline."""
    # Initialize clients
    milvus_uri = os.getenv("MILVUS_URI", "./milvus_demo.db")
    milvus_client = MilvusClient(uri=milvus_uri)
    groq_client = Groq()

    # Collection name
    collection_name = "my_rag_collection"

    # Generate a test embedding to get the dimension
    test_embedding = emb_text("This is a test", groq_client)
    embedding_dim = len(test_embedding)
    print(f"Embedding dimension: {embedding_dim}")
    print(f"Sample embedding values: {test_embedding[:10]}")

    # Set up Milvus collection
    setup_milvus_collection(milvus_client, collection_name, embedding_dim)

    # Process PDF and load data to Milvus
    pdf_path = os.getenv("PDF_FILE_PATH", "./pdf_files/WhatisMilvus.pdf")
    process_pdf_and_load_to_milvus(pdf_path, milvus_client, collection_name, groq_client)

    # Test the RAG pipeline
    question = "What is the Advanced Search Algorithms in Milvus?"
    answer = generate_rag_response(question, milvus_client, collection_name, groq_client)
    print(f"\nQuestion: {question}")
    print(f"Answer: {answer}")


if __name__ == "__main__":
    main()