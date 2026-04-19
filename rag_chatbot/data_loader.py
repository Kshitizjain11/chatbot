from llama_index.readers.file import PDFReader
from llama_index.core.node_parser import SentenceSplitter
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv() 

google_client = genai.Client()

EMBED_MODEL = "gemini-embedding-001"
EMBED_DIM = 768

splitter = SentenceSplitter(chunk_size=1000,chunk_overlap=200)

def load_and_chunk_pdf(path):
    docs = PDFReader().load_data(file=path)
    texts = [d.text for d in docs if getattr(d,"text",None)]

    chunks = []
    for t in texts:
        chunks.extend(splitter.split_text(t))
    
    return chunks

def embed_text(texts:list[str]) -> list[list[float]] :
    response = google_client.models.embed_content(
        model=EMBED_MODEL,
        contents=texts,
        config=types.EmbedContentConfig(output_dimensionality=768)
    )

    return [item.values for item in response.embeddings]
