import os
import openai
import ast
import re
from langchain_community.document_loaders import PyPDFLoader
from langchain.docstore.document import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


# Set OpenAI API key
openai.api_key = os.getenv("OPENAI_API_KEY")

def preprocess_text(text):
    """
    cleaning up spaces, tabs, and new lines
    """
    text = re.sub(r"\s+", " ", text)
    return text

def str_to_document(text: str):
    """
    extract metadata and content befortoring them into a dictionary
    """
    page_content_part, metadata_part = text.split(" metadata=")
    page_content = page_content_part.split("page_content=", 1)[1].strip("'")
    metadata = ast.literal_eval(metadata_part)

    return Document(page_content=page_content, metadata=metadata)

class PDFManager:
    """
    for psychodiagnostik
    """

    def __init__(self, path_name):
        self.path_name = path_name

    def load_pdf(self):
        """
        returns pdf split by pages
        """
        loader = PyPDFLoader(self.path_name)
        pages = loader.load_and_split()

        return pages

    def process_pdf(self, pages):
        """
        tokenizing before putting in chroma
        """
        text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            model_name="gpt-2",
            chunk_size=100, # 중요
            chunk_overlap=0, # 중요
        )

        raw_chunks = text_splitter.split_documents(pages)

        # Convert Document objects into strings
        chunks = [str(doc) for doc in raw_chunks]
        # Preprocess the text
        chunks = [preprocess_text(chunk) for chunk in chunks]
        # convert strings to Document objects
        docs = [str_to_document(chunk) for chunk in chunks]

        print("    Number of splitted tokens:", len(docs)) # row 개수

        return docs
