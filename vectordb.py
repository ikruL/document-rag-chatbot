
from io import BytesIO
from xmlrpc import client
import easyocr
from PIL import Image
import numpy as np
from pypdf import PdfReader
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
COLLECTION_NAME = "collections"
EMBED_MODEL = "qwen3-embedding:0.6b"


def extract_text_from_image(file_bytes: bytes):
    reader = easyocr.Reader(['vi', 'en'])
    try:
        image = Image.open(BytesIO(file_bytes))
        image_arr = np.array(image)
        result = reader.readtext(image_arr, detail=0)
        return "\n".join(result)
    except Exception as e:
        return f"Error: {str(e)}"


def get_db():
    embed_model = OllamaEmbeddings(model=EMBED_MODEL)
    vector_store = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embed_model,
        persist_directory="./chroma_db"
    )
    return vector_store, embed_model


def index_data(file_bytes: bytes, file_name: str, file_type: str):

    if file_type.startswith('image/'):
        texts = extract_text_from_image(file_bytes)
    elif file_type == 'application/pdf':
        reader = PdfReader(BytesIO(file_bytes))
        texts = ""
        for page in reader.pages:
            texts += page.extract_text() + "\n"
    else:
        return None, "Unsupported file type. Please upload a PDF or image file."
    if not texts.strip():
        return None, "Can't extract files"

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=200)
    documents = splitter.create_documents(
        texts=[texts],
        metadatas=[{
            "source": file_name,
            'type': file_type,
        }]
    )

    db, _ = get_db()
    db.add_documents(documents)

    return len(documents), f"Done! Extracted {file_name} Successfully !!"


vector_store, embed_model = get_db()
