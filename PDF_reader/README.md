# Chat with Multiple PDFs - Streamlit Application

## Overview
This application allows users to upload multiple PDF documents and interact with them using a conversational AI system. It processes the PDFs, extracts the text, splits the text into manageable chunks, and provides a way to ask questions and receive answers based on the documents' content.

## Features
- 📄 Upload multiple PDF documents.
- 📚 Extract and split text from uploaded PDFs for efficient processing.
- 💡 Use Sentence Transformer embeddings and FAISS for vector storage and similarity search.
- 🤖 Leverage GPT-Neo for natural language understanding and response generation.
- 💬 Maintain chat history for a natural conversational experience.

## Requirements

To run this application, you need the following:

- Python 3.x
- The following Python packages:
  - Streamlit
  - PyPDF2
  - Langchain
  - SentenceTransformers
  - FAISS
  - HuggingFaceHub API Key

## Installation

1. **Clone the repository:**
   ```bash
   git clone <repository_url>
   cd <repository_directory>
   ```

2. **Install the required dependencies:**
   ```bash
   pip install streamlit PyPDF2 langchain sentence-transformers faiss-cpu
   ```

3. **Set up your Hugging Face API key:**
   - Add your API key to the `config.py` file by replacing the placeholder `API_KEY` with your actual Hugging Face API key.

## How to Run

1. **Start the Streamlit app:**
   ```bash
   streamlit run app.py
   ```

2. **Upload your PDFs:**
   - Use the sidebar to upload PDF documents.
   - Click the **Process** button to extract text and create the vector store.

3. **Ask Questions:**
   - Once the documents are processed, input any question in the text field, and the system will answer based on the content of the PDFs.

4. **View Chat History:**
   - The application displays a history of the conversation for reference.

## Project Structure

- **`app.py`**: Contains the main application logic, including functions for text extraction, text chunking, vector storage, and the conversational AI system.
- **`config.py`**: Holds the API key required for Hugging Face integration.

## Key Functions

- **`get_pdf_text(pdf_documents)`**: Extracts text from uploaded PDF files.
- **`get_text_chunks(raw_text)`**: Splits the extracted text into smaller chunks for efficient processing.
- **`get_vectorstore(text_chunks)`**: Uses Sentence Transformer to create embeddings and stores them in a FAISS vector store for similarity search.
- **`get_conversational_chain(vector_store, API_KEY)`**: Sets up a conversational chain using the Hugging Face GPT-Neo model and integrates vector search with memory for contextual responses.

## Usage

1. **Upload PDFs**: Upload multiple PDFs through the sidebar and click **Process**.
2. **Ask Questions**: Type a question into the input field, and the app will generate a response based on the uploaded documents.
3. **View Chat History**: The conversation is displayed to provide context as you continue asking questions.

