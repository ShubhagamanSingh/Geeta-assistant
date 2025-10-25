import os
import pickle
from huggingface_hub import InferenceClient
from dotenv import load_dotenv

print("Starting vector store creation process using Hugging Face API...")

# Load environment variables
load_dotenv()

# Check for Hugging Face API Token
hf_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
if not hf_token:
    raise ValueError("Hugging Face API Token not found. Please set HUGGINGFACEHUB_API_TOKEN in the .env file.")

def simple_text_splitter(text, chunk_size=1000, chunk_overlap=100):
    """A basic function to split text into overlapping chunks."""
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size.")

    chunks = []
    start_index = 0
    while start_index < len(text):
        end_index = start_index + chunk_size
        chunks.append(text[start_index:end_index])
        start_index += chunk_size - chunk_overlap
    return chunks

def create_and_save_chunks(text_file="geeta_ocr.txt", chunks_file="hf_geeta_chunks.pkl"):
    """
    Reads a text file, splits it into chunks, and saves the chunks to a pickle file.
    The embeddings and FAISS index will be created in-memory by the Streamlit app.
    """
    try:
        with open(text_file, "r", encoding="utf-8") as f:
            text = f.read()
        print(f"Successfully loaded text from '{text_file}'.")
    except FileNotFoundError:
        print(f"Error: The file '{text_file}' was not found.")
        return

    # 1. Split the text into chunks
    chunks = simple_text_splitter(text)
    print(f"Text split into {len(chunks)} chunks.")

    # 2. Save the chunks to a pickle file
    with open(chunks_file, "wb") as f:
        pickle.dump(chunks, f)
    print(f"Chunks saved to '{chunks_file}'.")
    print("\nProcess complete. You can now upload the .pkl file to your GitHub repository.")


if __name__ == "__main__":
    create_and_save_chunks()