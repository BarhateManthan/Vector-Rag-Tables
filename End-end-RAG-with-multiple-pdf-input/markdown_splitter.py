import yaml
import os
from langchain.text_splitter import RecursiveCharacterTextSplitter

def load_config(path="config.yml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)

    """
    chunking:
        chunk_size: 800
        chunk_overlap: 100
        separators:
            - "\n## "
            - "\n### "
            - "\n- "
            - "\n\n"
            - " 
    """
class MarkdownChunker:
    def __init__(self, config):
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=config["chunking"]["chunk_size"],
            chunk_overlap=config["chunking"]["chunk_overlap"],
            separators=config["chunking"]["separators"]
        )

    def split(self, text):
        return self.splitter.split_text(text)
