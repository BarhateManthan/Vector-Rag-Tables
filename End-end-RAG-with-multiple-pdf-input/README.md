# 🧠 PDF Chatbot with Tables + Text using RAG & Gradio

This repo implements a **PDF-powered chatbot** that answers user queries using both structured tables and free-form text from PDF documents. It uses a **Retrieval-Augmented Generation (RAG)** pipeline with LangChain, OpenAI GPT, ChromaDB, and Gradio.

---

## 🚀 Features

- 📄 Extracts **tables** and **text** from PDFs via `pdfplumber`
- 🧩 Chunks combined content semantically using LangChain
- 🔍 Stores vector embeddings in **ChromaDB**
- 💬 Uses **GPT-4** to answer questions based on retrieved chunks
- 🧠 Maintains **chat history**
- 📑 Includes **page number metadata** for source traceability

---

## 📦 Setup

1. **Clone the repo**

```bash
git clone https://github.com/your-username/pdf-chatbot-rag.git
cd pdf-chatbot-rag
```

2. **Install dependencies**

```bash
pip install -r requirements.txt
```

3. **Configure your API key**



```
OPENAI_API_KEY=your-openai-api-key
```

---

## 🛠️ Usage

1. **Start the app**

```bash
python main.py
```

2. **In your browser**:
   - Upload a PDF with **tables and/or text**
   - Wait for preprocessing
   - Ask any question — including follow-ups!

---

## 📁 Project Structure

```
.
├── app.py                   # Gradio app + user interface
├── rag_pipeline.py          # RAG logic (embedding, retrieval, generation)
├── table_loader.py          # Extracts and formats tables
├── text_loader.py           # Extracts page-wise text
├── markdown_splitter.py     # Markdown-based chunking logic
├── logger.py                # Logging utility
├── config.yml               # Chunking configuration
└── README.md
```

---

## 🧱 Architecture Overview

### Key Design Decisions & Trade-offs

| Component                     | Decision / Rationale                                      |
|------------------------------|------------------------------------------------------------|
| PDF Parsing                  | Used `pdfplumber`                                          |
| Text & Table Unification     | Markdown format provides semantic structure & readability  |
| ChromaDB                     | persistent vector store for local RAG                      |       
| Chunking Strategy            | Recursive splitting by heading → preserves context         |
| Metadata Usage               | Enables page-level references in GPT answers               |

---

## 🧩 Implementation Approach

### Solving the Multi-Modal (Images are not considered) Challenge

PDFs contain **structured** (tables) and **unstructured** (text) data. Our approach:

- Uses `pdfplumber` to extract **tables** → convert to markdown with `pandas.to_markdown()`
- Uses `pdfplumber` to extract **text** per page → add metadata (e.g., "Page 3")
- Combines both into a **single markdown document**
- Splits that into manageable **semantic chunks** using `RecursiveCharacterTextSplitter`
- Stores those chunks in Chroma with **metadata: `page`, `type`**
- GPT is prompted with chat **history + context** and instructed to cite **page numbers**

---


---

## ⚠️ Limitations & Future Work

### Known Issues

- ❌ Currently, pdfplumber processes pages sequentially. We can speed this up by using concurrent.futures or joblib
- ❌ Filter uninformative pages
- ❌ No OCR for scanned image-based PDFs
- ⚠️ Limited support for very large PDFs (>100 pages)
- ❌ No multi-user session state which also be done my making changes in app.py file by using session ID and keeping history based on session ID 

### Potential Improvements

- 🔄 Add **OCR fallback** with Tesseract for image PDFs
- 🔍 Show **inline citations** in UI or model reply
- 💾 Use **embedding cache** to skip reprocessing
- 👥 Enable **multi-user sessions** in Gradio or FastAPI backend
- 📊 Add UI section to **visualize extracted tables** directly

---

## 💡 Example Interaction

> **User**: Trequite site is located at what coordinates?  
> **Assistant**: The Trequite site is located at coordinates N 50.425629, W -4.369987. (Source 1, Page 33)  
> **User**: which site I was taling about in the previous question.
> **Assistant**: In the previous question, you were talking about the Trequite site.

---

## ✅ License

MIT — free for academic and commercial use.
