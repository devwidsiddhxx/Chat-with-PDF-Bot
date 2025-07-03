# 💬 Chat with PDF using LangChain, Hugging Face & OpenRouter

An intelligent, fully local PDF Question-Answering app built with **LangChain**, **FAISS**, **Hugging Face Embeddings**, and **OpenRouter (ChatGPT)**. Just upload your PDF(s), ask questions, and get context-aware answers instantly.

---

## 📌 Features

- ✅ Extracts and indexes PDF content for question-answering
- ✅ Uses `all-MiniLM-L6-v2` from Hugging Face for fast, lightweight embeddings
- ✅ Stores vector embeddings locally using **FAISS**
- ✅ Semantic search for most relevant chunks
- ✅ ChatGPT (via OpenRouter) for accurate answers,Groq
- ✅ Clean and responsive UI with **Streamlit**
- ✅ Works entirely offline for retrieval (only API call is for LLM)

---

## 🚀 Quick Demo

https://drive.google.com/drive/folders/1ILiz_ahabnkXM7kFYIWApgEjSlzNrNGU?usp=sharing

---

## 🧠 How It Works

1. **Upload PDFs**
   - Extracts and concatenates raw text using `PyPDF2`.

2. **Text Chunking**
   - Splits text using LangChain's `RecursiveCharacterTextSplitter` (4,000 char chunks, 400 overlap).

3. **Embedding**
   - Generates sentence-level embeddings via `all-MiniLM-L6-v2`.

4. **Store in FAISS**
   - Embeddings are stored in a local FAISS vector index.

5. **Ask a Question**
   - Your question is semantically compared to stored chunks using FAISS similarity search.

6. **Answer Generation**
   - Most relevant chunks + your question are sent to `gpt-3.5-turbo` via **OpenRouter**.
   - The model returns an accurate, context-aware response.

---

## 🛠️ Tech Stack

| Layer       | Technology                              |
|-------------|------------------------------------------|
| UI          | Streamlit                               |
| PDF Parsing | PyPDF2                                   |
| NLP Engine  | LangChain                                |
| Embeddings  | Hugging Face: `all-MiniLM-L6-v2`         |
| Vector DB   | FAISS                                    |
| LLM         | OpenRouter (`gpt-3.5-turbo`)             |
| Config      | Python-dotenv for secure API key loading |

---

## 📂 Folder Structure

```bash
.
├── app.py                # Main Streamlit app
├── faiss_index/         # Stored vector database
├── .env                 # API keys (not committed)
├── requirements.txt     # All dependencies
└── README.md            # This file

Feel free to contribute to this project by improving features, fixing bugs, or optimizing performance.  
Built  by  Siddharth Venkatesh.
