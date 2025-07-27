import gradio as gr
from text_loader import extract_text_chunks
from table_loader import extract_table_chunks
from markdown_splitter import load_config, MarkdownChunker
from rag_pipeline import RAGPipeline
from logger import get_logger
from langchain.schema import Document

config = load_config()
pipeline = None
logger = get_logger("PDF Chatbot")

def init_pipeline(pdf):
    # Extract annotated text and table chunks (each with page, type, text)
    
    text_chunks = extract_text_chunks(pdf.name)
    table_chunks, total_tables = extract_table_chunks(pdf.name)
    all_chunks = text_chunks + table_chunks

    logger.info(f"Total extracted chunks: {len(all_chunks)}, including {total_tables} tables.")

    # Sort chunks by page to preserve order
    all_chunks.sort(key=lambda c: c["page"])

    # Create Document objects with metadata
    
    docs = []
    for chunk in all_chunks:
        docs.append(Document(
            page_content=chunk["text"],
            metadata={
                "page": chunk["page"],
                "type": chunk["type"]
            }))

    # Build RAG pipeline
    global pipeline
    pipeline = RAGPipeline(docs, model=config["openai"]["model"])
    return "PDF processed. Chatbot is ready to answer your questions."

def chat_with_pdf(message, history):
    if not pipeline:
        return "Please upload and process a PDF first."
    return pipeline.generate(message, history)

# Gradio interface
with gr.Blocks(title="PDF Chatbot with Tables + Text") as demo:
    chatbot = gr.ChatInterface(fn=chat_with_pdf)

    with gr.Row():
        pdf_input = gr.File(label="Upload PDF")
        upload_btn = gr.Button("Process PDF")
        status = gr.Textbox(label="Status", interactive=False)

    upload_btn.click(fn=init_pipeline, inputs=[pdf_input], outputs=[status])

demo.launch(share=True)
