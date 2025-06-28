import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.chat_models import ChatOpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
from pathlib import Path
import time

load_dotenv()

# Configuration - set these in your .env file
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=4000,
        chunk_overlap=400
    )
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

def get_conversational_chain():
    prompt_template = """
    Answer the question from the provided context. If the answer is not in the context, say "I cannot find the answer in the provided documents."
    
    Context:\n {context}\n
    Question: \n{question}\n
    Answer:
    """
    
    # Use a model that's available on OpenRouter
    # Popular options: meta-llama/llama-3.1-8b-instruct:free, microsoft/phi-3-mini-128k-instruct:free
    model = ChatOpenAI(
        model_name="meta-llama/llama-3.1-8b-instruct:free",  # Free model available on OpenRouter
        temperature=0.3,
        openai_api_key=OPENROUTER_API_KEY,
        openai_api_base="https://openrouter.ai/api/v1",  # Use openai_api_base instead of base_url
        max_retries=3,
        request_timeout=120
    )
    
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

def user_input(user_question):
    if not (Path("faiss_index") / "index.faiss").exists():
        st.warning("Please upload and process PDF files first.")
        return
    
    try:
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        new_db = FAISS.load_local(
            "faiss_index", 
            embeddings, 
            allow_dangerous_deserialization=True
        )
        
        # Get more relevant documents
        docs = new_db.similarity_search(user_question, k=4)
        chain = get_conversational_chain()
        
        time.sleep(1)  # Rate limiting for free tier
        
        with st.spinner("Generating answer..."):
            response = chain(
                {"input_documents": docs, "question": user_question},
                return_only_outputs=True
            )
            st.write("Reply: ", response["output_text"])
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        st.info("If you're getting model errors, try switching to a different model in the code or check OpenRouter's available models.")

def main():
    st.set_page_config("Chat PDF", page_icon="💁")
    st.header("Chat with PDF 💁")
    
    # Add model info
    st.info("Using Llama 3.1 8B model via OpenRouter (free tier)")

    user_question = st.text_input("Ask a Question from the PDF Files")

    if user_question:
        user_input(user_question)

    with st.sidebar:
        st.title("Menu:")
        pdf_docs = st.file_uploader(
            "Upload your PDF Files and Click on the Submit & Process Button", 
            accept_multiple_files=True,
            type=["pdf"]
        )
        if st.button("Submit & Process"):
            if not pdf_docs:
                st.warning("Please upload at least one PDF file.")
            elif not OPENROUTER_API_KEY:
                st.error("Please set your OPENROUTER_API_KEY in the .env file.")
            else:
                with st.spinner("Processing..."):
                    try:
                        raw_text = get_pdf_text(pdf_docs)
                        if not raw_text.strip():
                            st.warning("No text could be extracted from the PDF files. Please check if they contain readable text.")
                        else:
                            text_chunks = get_text_chunks(raw_text)
                            get_vector_store(text_chunks)
                            st.success("Processing complete! You can now ask questions.")
                    except Exception as e:
                        st.error(f"Error processing PDFs: {str(e)}")

if __name__ == "__main__":
    main()
