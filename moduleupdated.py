import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.chat_models import ChatOpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from pathlib import Path
import time
import requests
import json

# Embedded API configuration - No setup required!
GROQ_API_KEY = "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"  # Free Groq API
GROQ_BASE_URL = "https://api.groq.com/openai/v1"

st.set_page_config("Chat PDF - Ready to Use", page_icon="💁")

@st.cache_resource
def load_embeddings():
    """Load embeddings model"""
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'}
    )

def get_pdf_text(pdf_docs):
    """Extract text from PDF files"""
    text = ""
    for pdf in pdf_docs:
        try:
            pdf_reader = PdfReader(pdf)
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
        except Exception as e:
            st.error(f"Error reading PDF {pdf.name}: {str(e)}")
    return text

def get_text_chunks(text):
    """Split text into chunks"""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=2000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks):
    """Create vector store from text chunks"""
    embeddings = load_embeddings()
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")
    return vector_store

def call_groq_api(prompt, context):
    """Call Groq API directly for better reliability"""
    try:
        headers = {
            "Authorization": f"Bearer {GROQ_API_KEY}",
            "Content-Type": "application/json"
        }
        
        full_prompt = f"""Based on the following context, answer the question. If the answer is not in the context, say "I don't have enough information to answer this question based on the provided documents."

Context: {context}

Question: {prompt}

Answer:"""
        
        data = {
            "model": "llama3-8b-8192",
            "messages": [
                {"role": "user", "content": full_prompt}
            ],
            "temperature": 0.3,
            "max_tokens": 1000
        }
        
        response = requests.post(
            f"{GROQ_BASE_URL}/chat/completions",
            headers=headers,
            json=data,
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            return result['choices'][0]['message']['content']
        else:
            # Fallback to simple keyword matching
            return simple_answer_extraction(context, prompt)
            
    except Exception as e:
        st.warning(f"API call failed, using fallback method: {str(e)}")
        return simple_answer_extraction(context, prompt)

def simple_answer_extraction(context, question):
    """Fallback method using keyword matching"""
    # Convert to lowercase for matching
    context_lower = context.lower()
    question_lower = question.lower()
    
    # Extract key terms from question
    question_words = [word for word in question_lower.split() if len(word) > 3]
    
    # Find the most relevant sentences
    sentences = context.split('.')
    relevant_sentences = []
    
    for sentence in sentences:
        sentence_lower = sentence.lower()
        score = 0
        for word in question_words:
            if word in sentence_lower:
                score += 1
        
        if score > 0:
            relevant_sentences.append((sentence.strip(), score))
    
    # Sort by relevance score
    relevant_sentences.sort(key=lambda x: x[1], reverse=True)
    
    if relevant_sentences:
        # Return top 3 most relevant sentences
        answer = ". ".join([sent[0] for sent in relevant_sentences[:3]])
        return f"Based on the document: {answer}"
    else:
        return "I couldn't find specific information about that in the provided documents. Please try rephrasing your question."

def user_input(user_question):
    """Process user question and return answer"""
    if not (Path("faiss_index") / "index.faiss").exists():
        st.warning("Please upload and process PDF files first.")
        return
    
    try:
        # Load the vector store
        embeddings = load_embeddings()
        new_db = FAISS.load_local(
            "faiss_index", 
            embeddings, 
            allow_dangerous_deserialization=True
        )
        
        # Search for relevant documents
        docs = new_db.similarity_search(user_question, k=3)
        
        # Combine context from retrieved documents
        context = "\n".join([doc.page_content for doc in docs])
        
        with st.spinner("Generating answer..."):
            # Call the API or fallback method
            answer = call_groq_api(user_question, context)
            
            st.write(answer)
            
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        st.info("Please try uploading the PDF files again.")

def main():
    st.header("💁 Chat with PDF")
    
    # Main chat interface
    user_question = st.text_input(
        "Ask a Question about your PDFs:", 
        placeholder="e.g., What is the main topic? Who are the key people mentioned?"
    )

    if user_question:
        user_input(user_question)

    # Sidebar for file upload
    with st.sidebar:
        st.title("📁 Upload Documents")
        
        pdf_docs = st.file_uploader(
            "Choose PDF Files:", 
            accept_multiple_files=True,
            type=["pdf"],
            help="Upload one or more PDF files to analyze"
        )
        
        if st.button("🚀 Process Documents", use_container_width=True):
            if not pdf_docs:
                st.warning("Please upload at least one PDF file.")
            else:
                with st.spinner("Processing your documents..."):
                    # Progress tracking
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    # Extract text
                    status_text.text("📖 Reading PDF files...")
                    progress_bar.progress(25)
                    raw_text = get_pdf_text(pdf_docs)
                    
                    if raw_text.strip():
                        # Split into chunks
                        status_text.text("✂️ Processing text...")
                        progress_bar.progress(50)
                        text_chunks = get_text_chunks(raw_text)
                        
                        # Create vector store
                        status_text.text("🔍 Building search index...")
                        progress_bar.progress(75)
                        get_vector_store(text_chunks)
                        
                        progress_bar.progress(100)
                        status_text.text("✅ Ready!")
                        
                        st.success(f"Successfully processed {len(pdf_docs)} PDF file(s)!")
                        
                        # Document statistics
                        st.markdown("### 📊 Processing Summary")
                        st.write(f"📄 **Files:** {len(pdf_docs)}")
                        st.write(f"📝 **Text chunks:** {len(text_chunks)}")
                        st.write(f"📏 **Total text:** {len(raw_text):,} characters")
                        
                        st.balloons()  # Celebration animation
                        
                    else:
                        st.error("❌ Could not extract text from the PDF files. Please ensure they contain readable text.")
        
        st.markdown("---")
        
        st.markdown("### 🚀 How to Use")
        st.markdown("""
        1. **Upload** your PDF files above
        2. **Click** 'Process Documents' 
        3. **Ask** questions in the main area
        4. **Get** instant answers!
        """)

if __name__ == "__main__":
    main()
