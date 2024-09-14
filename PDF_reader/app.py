import streamlit as st
from config import API_KEY
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter 
from langchain.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains.conversational_retrieval.base import ConversationalRetrievalChain
from langchain.llms.huggingface_hub import HuggingFaceHub
from sentence_transformers import SentenceTransformer

API_KEY = API_KEY

# Function to extract text from PDF documents
def get_pdf_text(pdf_documents):
    text = ""
    for pdf in pdf_documents:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text 

# Function to split text into smaller chunks
def get_text_chunks(raw_text):
    text_splitter = CharacterTextSplitter(
        separator='\n',
        chunk_size=1000,
        chunk_overlap=200
    )
    chunks = text_splitter.split_text(raw_text)
    return chunks

# Function to create vector store using Sentence Transformer for embeddings
def get_vectorstore(text_chunks):
    # Use the sentence transformer model for embeddings
    model = SentenceTransformer('hkunlp/instructor-xl')
    embeddings = [model.encode(chunk) for chunk in text_chunks]
    
    # Create FAISS vector store from the embeddings
    vector_store = FAISS.from_texts(texts=text_chunks, embeddings=embeddings)
    return vector_store

# Function to set up the conversational chain
def get_conversational_chain(vector_store, API_KEY):
    # Using HuggingFaceHub for the LLM (gpt-neo model)
    llm = HuggingFaceHub(
        repo_id="EleutherAI/gpt-neo-2.7B",
        model_kwargs={"temperature": 0.5, "max_length": 512},
        huggingfacehub_api_token=API_KEY
    )
    
    # Setting up memory for conversation history
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
    
    # Create conversational retrieval chain
    conversational_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vector_store.as_retriever(),
        memory=memory
    )
    
    return conversational_chain

def main():
    st.set_page_config(page_title='Chat with multiple PDFs', page_icon=':books:')

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    st.header('Chat with multiple PDFs :books:')
    user_question = st.text_input('Ask any question from the document')

    # Sidebar for uploading PDFs
    with st.sidebar:
        st.subheader('Your Documents')
        pdf_documents = st.file_uploader('Upload your PDFs here and click on process', accept_multiple_files=True)

        # Processing the PDFs
        if st.button('Process'):
            with st.spinner("Processing..."):
                # Extract text from the PDFs
                raw_text = get_pdf_text(pdf_documents)

                # Split text into chunks
                text_chunks = get_text_chunks(raw_text)

                # Create vector store using SentenceTransformer embeddings
                vector_store = get_vectorstore(text_chunks)

                # Set up conversational chain with HuggingFaceHub
                st.session_state.conversation = get_conversational_chain(vector_store, API_KEY)
                st.success("Documents processed. You can now ask questions!")

    # Generating responses based on the user's question
    if st.session_state.conversation and user_question:
        with st.spinner("Generating response..."):
            response = st.session_state.conversation({'question': user_question})
            st.session_state.chat_history = response['chat_history']

        # Displaying the conversation history
        st.subheader("Chat History")
        for message in st.session_state.chat_history:
            if message['role'] == 'user':
                st.write(f"**User**: {message['content']}")
            else:
                st.write(f"**Bot**: {message['content']}")

if __name__ == '__main__':
    main()
