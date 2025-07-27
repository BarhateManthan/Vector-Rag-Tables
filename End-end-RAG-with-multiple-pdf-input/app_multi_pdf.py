import gradio as gr
from text_loader import extract_text_chunks
from table_loader import extract_table_chunks
from markdown_splitter import load_config
from rag_pipeline import RAGPipeline
from logger import get_logger
from langchain.schema import Document
import os

config = load_config()
pipeline_map = {}  # filename (stem) -> RAGPipeline
logger = get_logger("Auto-Select PDF Chatbot")

def init_pipelines(file_list):
    responses = []

    for file in file_list:
        pdf_path = file.name
        pdf_name = os.path.splitext(os.path.basename(pdf_path))[0]

        # Extract content
        text_chunks = extract_text_chunks(pdf_path)
        table_chunks, total_tables = extract_table_chunks(pdf_path)
        all_chunks = text_chunks + table_chunks
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

        # Create pipeline with unique DB per PDF
        persist_dir = f"chroma_store/{pdf_name}"
        pipeline_map[pdf_name] = RAGPipeline(docs, model=config["openai"]["model"], persist_dir=persist_dir)
        responses.append(f" {pdf_name} processed with {len(all_chunks)} chunks.")

    return "\n".join(responses)

def chat_with_pdf(message, history):
    best_match = None
    best_score = float("-inf")
    best_answer = None

    for name, pipeline in pipeline_map.items():
        context = pipeline.retrieve(message)
        if not context:
            continue
        prompt = f"{'\n'.join([doc.page_content for doc in context])}\n\nQuestion: {message}\nAnswer:"
        relevance = sum([len(doc.page_content) for doc in context])
        if relevance > best_score:
            best_score = relevance
            best_match = name
            best_answer = pipeline.generate(message, history)

    if best_answer:
        return f"📄 Answer from **{best_match}**:\n\n{best_answer}"
    return "No relevant information found in any PDF."

# Gradio interface
with gr.Blocks(title="Auto-PDF RAG Chatbot") as demo:
    chatbot = gr.ChatInterface(fn=chat_with_pdf)
    with gr.Row():
        pdf_input = gr.File(file_types=[".pdf"], file_count="multiple", label="Upload PDFs")
        upload_btn = gr.Button("Process PDFs")
        status = gr.Textbox(label="Status", interactive=False)

    upload_btn.click(fn=init_pipelines, inputs=[pdf_input], outputs=[status])

demo.launch(share=True)