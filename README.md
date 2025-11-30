
# KnowledgeBase Agent (OpenRouter + FAISS + Streamlit)

This is a Chat-style KnowledgeBase Agent designed to answer questions from uploaded company documents (PDF/TXT).  
It uses:
- **OpenRouter** for LLM responses & embeddings  
- **FAISS** for vector search  
- **Streamlit** for the interactive chat UI  
- **Zero LangChain / Zero Chroma** → No dependency conflicts on Streamlit Cloud  

## 🚀 Features
- Upload multiple PDF/TXT documents  
- Ingest documents (extract, chunk, embed, store into FAISS)  
- Chat with your documents  
- Persistent FAISS index saved on disk  
- Fast and lightweight  

## 🧰 Tech Stack
- **Frontend:** Streamlit  
- **LLM & Embeddings:** OpenRouter  
- **Vector DB:** FAISS (persistent)  
- **Document Processing:** PyPDF + custom chunking  

## 📦 Installation (Local)
```
pip install -r requirements.txt
```

Create `.env` file:
```
OPENROUTER_API_KEY="your-key"
CHROMA_PERSIST_DIR="./faiss_data"
```

Ingest documents:
```
python ingest.py
```

Run app:
```
streamlit run streamlit_app.py
```

## 🌐 Deployment (Streamlit Cloud)
- Upload files to GitHub  
- Set main file: `streamlit_app.py`  
- Add secrets (TOML format):
```
OPENROUTER_API_KEY="sk-or-xxxx"
CHROMA_PERSIST_DIR="./faiss_data"
```

## 📁 Project Structure
```
📦 KnowledgeBase-Agent
 ┣ streamlit_app.py
 ┣ agent.py
 ┣ ingest.py
 ┣ requirements.txt
 ┣ README.md
 ┣ architecture.txt
 ┣ example_config.env
 ┣ docs/
 ┗ faiss_data/
```

## ❤️ Credits
Built by Chandana with ChatGPT assistance.
