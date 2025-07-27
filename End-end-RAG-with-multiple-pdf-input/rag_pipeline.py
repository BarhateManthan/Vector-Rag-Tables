import os
from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain_openai import AzureOpenAIEmbeddings
from langchain.schema import Document
from openai import AzureOpenAI
from logger import get_logger
import shutil

openai_api_key = "YOUR_AZURE_OPENAI_API_KEY"

logger = get_logger("RAG")

class RAGPipeline:
    def __init__(self, docs, model="gpt-4o", persist_dir="chroma_store"):
        self.model = model
        self.client = AzureOpenAI(
            api_key=openai_api_key,
            api_version="Your api version here",
            azure_endpoint="Your endpoint here"
        )

        self.persist_dir = persist_dir
        # Use Azure OpenAI embeddings instead of regular OpenAI
        self.embeddings = AzureOpenAIEmbeddings(
            azure_deployment="text-embedding-3-small",  # Replace with your embedding deployment name
            azure_endpoint="Your endpoint here",
            api_key=openai_api_key,
            api_version="Your api version here"
        )

        if os.path.exists(persist_dir):
            shutil.rmtree(persist_dir) # cleaning existing db optional 

        logger.info("Building Chroma vector store...")
        self.db = Chroma.from_documents(
            documents=docs,
            embedding=self.embeddings,
            persist_directory=self.persist_dir
        )
        self.db.persist()
        logger.info("Chroma store built.")

    def retrieve(self, query, top_k=3):
        results = self.db.similarity_search(query, k=top_k)
        return results

    def generate(self, query, history=None):
        
        results = self.retrieve(query)

       #retrieved context chunks
        context_blocks = []
        for i, doc in enumerate(results, start=1):
            page = doc.metadata.get("page", "?")
            dtype = doc.metadata.get("type", "text")
            context_blocks.append(f"### Source {i} (Page {page}, Type: {dtype})\n{doc.page_content}")

       #Format conversation history
        history_text = ""
        if history:
            for user_msg, assistant_reply in history:
                history_text += f"User: {user_msg}\nAssistant: {assistant_reply}\n"

        # Step 3: Construct prompt
        context_text = '\n'.join(context_blocks)
        
        prompt = f"""You are an expert in handling data for Clinical Trials particularly in ADSL and SDTM standards. 
Use the following context and prior conversation history to answer the user's latest question clearly and concisely. 
Always ground your answer in the given context. Do not make up any facts. 
When applicable, cite the page number of the source (e.g., "According to Page 3…" or "(see Page 2)".


### Conversation History
{history_text}

### Context
{context_text}

### Current Question
{query}

### Answer
"""

        logger.info(f"Calling Azure OpenAI with {len(context_blocks)} chunks and history length: {len(history) if history else 0}")
        
        completion = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3
        )
        
        return completion.choices[0].message.content
