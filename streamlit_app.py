import os
import pickle
import streamlit as st # type: ignore
import faiss # type: ignore
import numpy as np # type: ignore
import re
from huggingface_hub import InferenceClient # type: ignore
from dotenv import load_dotenv # type: ignore

# --- Load environment variables ---
load_dotenv()

# --- Streamlit Page Configuration ---
st.set_page_config(
    page_title="Geeta Assistant",
    page_icon="🕉️",
    layout="centered"
)

# --- Custom Styling ---
st.markdown("""
<style>
    .stApp {
        background-color: black;
    }
    .st-chat-message-user {
        background-color: #e8f5e8;
        padding: 10px;
        border-radius: 10px;
        margin: 5px 0;
    }
    .st-chat-message-assistant {
        background-color: #fff3cd;
        padding: 10px;
        border-radius: 10px;
        margin: 5px 0;
    }
    h1 {
        color: #2e7d32;
        text-align: center;
        font-family: 'Georgia', serif;
    }
</style>
""", unsafe_allow_html=True)

# --- Text Cleaning Functions ---
def is_meaningful_text(text, min_meaningful_chars=10):
    """Check if text contains meaningful content"""
    if not text or len(text.strip()) < min_meaningful_chars:
        return False
    
    # Remove special characters and check if enough text remains
    clean_text = re.sub(r'[^a-zA-Z0-9\u0900-\u097F\s]', '', text)
    clean_text = clean_text.strip()
    
    # Check if it's mostly special characters or garbage
    if len(clean_text) < min_meaningful_chars:
        return False
        
    # Check for repetitive patterns (like ~~~ ^~ ^~)
    repetitive_pattern = re.search(r'(.)\1{3,}', text)
    if repetitive_pattern:
        return False
        
    return True

def clean_chunk_text(text):
    """Clean and extract meaningful parts from text chunk"""
    lines = text.split('\n')
    meaningful_lines = []
    
    for line in lines:
        clean_line = line.strip()
        if is_meaningful_text(clean_line):
            meaningful_lines.append(clean_line)
    
    return ' '.join(meaningful_lines[:3])  # Return first 3 meaningful lines

# --- Helper: Embedding Generation ---
def get_embeddings(texts, model_name="sentence-transformers/all-MiniLM-L6-v2"):
    """
    Generate embeddings using the InferenceClient
    """
    try:
        client = InferenceClient(token=os.getenv("HUGGINGFACEHUB_API_TOKEN"))
        embeddings = client.feature_extraction(text=texts, model=model_name)
        return embeddings
    except Exception as e:
        st.error(f"❌ Embedding generation failed: {e}")
        # Return random embeddings as fallback
        return np.random.rand(len(texts), 384).tolist()

# --- Cache the initialization ---
@st.cache_resource
def load_chatbot_components():
    try:
        with open("hf_geeta_chunks.pkl", "rb") as f:
            chunks = pickle.load(f)

        st.info(f"Loaded {len(chunks)} chunks from Bhagavad Gita")
        
        # Clean the chunks
        cleaned_chunks = []
        for i, chunk in enumerate(chunks):
            cleaned = clean_chunk_text(str(chunk))
            if cleaned and len(cleaned) > 20:  # Only keep meaningful chunks
                cleaned_chunks.append(cleaned)
        
        st.info(f"After cleaning: {len(cleaned_chunks)} meaningful chunks remaining")
        
        if len(cleaned_chunks) == 0:
            st.error("No meaningful text found in chunks. Please check your data file.")
            return None, None

        st.info("Creating FAISS index in memory. Please wait...")

        # Process embeddings in smaller batches
        batch_size = 5
        all_embeddings = []
        
        progress_bar = st.progress(0)
        for i in range(0, len(cleaned_chunks), batch_size):
            batch = cleaned_chunks[i:i + batch_size]
            batch_embeddings = get_embeddings(batch)
            all_embeddings.extend(batch_embeddings)
            
            # Update progress
            progress = min((i + batch_size) / len(cleaned_chunks), 1.0)
            progress_bar.progress(progress)

        embeddings_array = np.array(all_embeddings, dtype=np.float32)
        
        # Ensure embeddings are 2D
        if len(embeddings_array.shape) == 1:
            embeddings_array = np.expand_dims(embeddings_array, axis=0)
        
        # Create FAISS index
        index = faiss.IndexFlatL2(embeddings_array.shape[1])
        index.add(embeddings_array)

        st.success("✅ FAISS index created successfully!")
        return cleaned_chunks, index

    except FileNotFoundError:
        st.error("⚠️ 'hf_geeta_chunks.pkl' file not found. Place it in the same folder.")
        return None, None
    except Exception as e:
        st.error(f"❌ Initialization failed: {e}")
        return None, None

# --- Emotional Support Responses ---
def get_emotional_support_response(query, context):
    """
    Provide specialized emotional support responses for difficult situations
    """
    query_lower = query.lower()
    
    # Emotional support patterns
    emotional_keywords = {
        "downfall": "O dear one, even Arjuna felt despair on the battlefield. Remember: 'No one who does good work will ever come to a bad end, either here or in the world to come.' (Bhagavad Gita)",
        "sad": "The Gita teaches that happiness and distress come and go like seasons. They are born of sense contact and one must learn to tolerate them without being disturbed.",
        "depress": "When depression clouds your vision, remember Krishna's words: 'From the world of the senses, Arjuna, comes heat and comes cold, and pleasure and pain. They come and they go forever; they are transient. Arise above them, strong soul.'",
        "hope": "Have hope, dear seeker. Krishna says: 'To those who are constantly devoted and worship Me with love, I give the understanding by which they can come to Me.'",
        "failure": "What you call failure is but a bend in the river of life. The Gita says: 'You have a right to perform your prescribed duty, but you are not entitled to the fruits of action.' Perform your duty without attachment to success or failure.",
        "stress": "In times of stress, practice equanimity. Krishna advises: 'A person is said to be established in self-realization and is called a yogi when he is fully satisfied by virtue of acquired knowledge and realization.'",
        "anxiety": "For anxiety, the Gita prescribes: 'Therefore, without being attached to the fruits of activities, one should act as a matter of duty, for by working without attachment one attains the Supreme.'",
        "lost": "When you feel lost, remember: 'Whenever there is a decline in righteousness and an increase in unrighteousness, O Arjuna, at that time I manifest Myself on earth.' The divine is always with you.",
        "confus": "In confusion, seek clarity through knowledge. Krishna says: 'When your mind has overcome the confusion of duality, you will attain the state of perfect yoga.'",
        "trouble": "In troubled times, remember: 'The soul can never be cut to pieces by any weapon, nor burned by fire, nor moistened by water, nor withered by the wind.' Your true self is eternal and indestructible."
    }
    
    # Check for emotional keywords
    for keyword, response in emotional_keywords.items():
        if keyword in query_lower:
            return response
    
    # General emotional support
    return f"""O dear seeker, I feel your pain and understand your struggle. 

In the Bhagavad Gita, Lord Krishna teaches Arjuna who was also in deep despair:

"Never was there a time when I did not exist, nor you, nor all these kings; nor in the future shall any of us cease to be."

This too shall pass, my friend. The soul is eternal, and your current difficulties are but temporary circumstances. 

Have faith, perform your duty without attachment to results, and remember that every experience - both pleasant and painful - serves your spiritual growth.

What specific aspect of your situation troubles you most?"""

# --- RAG Logic ---
def get_geeta_response(query, chunks, index, k=5):  # Increased k to get more context options
    try:
        query_embedding = get_embeddings([query])
        query_embedding = np.array(query_embedding, dtype=np.float32)
        
        # Ensure query embedding is 2D
        if len(query_embedding.shape) == 1:
            query_embedding = np.expand_dims(query_embedding, axis=0)
    except Exception as e:
        st.error(f"❌ Failed to generate embedding: {e}")
        return get_emotional_support_response(query, "")

    try:
        # Get more chunks to have better options
        distances, indices = index.search(query_embedding, k)
        
        # Filter to get only meaningful context
        meaningful_contexts = []
        for i in indices[0]:
            if i < len(chunks):  # Safety check
                chunk_text = chunks[i]
                if is_meaningful_text(chunk_text, min_meaningful_chars=15):
                    meaningful_contexts.append(chunk_text)
        
        # Use meaningful contexts or fallback
        if meaningful_contexts:
            context = "\n\n".join(meaningful_contexts[:3])  # Use top 3 meaningful contexts
        else:
            context = "The Bhagavad Gita teaches eternal truths about duty, devotion, and the nature of reality."
        
        # For emotional queries, use specialized emotional support
        emotional_words = ["downfall", "sad", "depress", "hope", "failure", "stress", "anxiety", "lost", "confus", "trouble"]
        if any(word in query.lower() for word in emotional_words):
            return get_emotional_support_response(query, context)
        
        # Try to use Hugging Face model for non-emotional queries
        try:
            client = InferenceClient(token=os.getenv("HUGGINGFACEHUB_API_TOKEN"))
            
            prompt = f"""As Lord Krishna, answer this spiritual question based on the Bhagavad Gita.

Context: {context[:800]}

Question: {query}

Answer as Krishna with divine wisdom and compassion:"""
            
            response = client.text_generation(
                prompt,
                model="google/flan-t5-base",
                max_new_tokens=200,
                temperature=0.7
            )
            
            if response and len(response.strip()) > 20:
                return response
        
        except Exception as e:
            st.warning(f"AI model unavailable, using wisdom-based response")
        
        # Fallback response
        return f"""O seeker, the wisdom of the Bhagavad Gita illuminates your path. 

While specific verses may vary, the eternal truth remains: perform your duty without attachment, offer all actions to the divine, and maintain equanimity in success and failure.

Your current situation, though challenging, is part of your spiritual journey. Trust in the divine plan and continue walking the path of righteousness.

What specific guidance do you seek from the Gita today?"""
        
    except Exception as e:
        st.error(f"❌ Retrieval failed: {e}")
        return get_emotional_support_response(query, "")

# --- Streamlit App Layout ---
st.title("🕉️ Geeta Assistant 🕉️")
st.markdown("### Your personal guide to Bhagavad Gita wisdom")

# Ensure token exists
if not os.getenv("HUGGINGFACEHUB_API_TOKEN"):
    st.error("⚠️ Missing Hugging Face API key in your .env file.")
    st.info("Get a free token from: https://huggingface.co/settings/tokens")
    st.stop()

chunks, index = load_chatbot_components()

if all([chunks is not None, index is not None]):
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "O seeker, I am Krishna. Ask me anything about life, duty, spirituality, or the wisdom of the Bhagavad Gita. I am here to guide you through both joyful and difficult times."}
        ]

    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat input
    if prompt := st.chat_input("What wisdom do you seek from the Gita?"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Krishna is contemplating your question..."):
                answer = get_geeta_response(prompt, chunks, index)
                st.markdown(answer)
                st.session_state.messages.append({"role": "assistant", "content": answer})
else:
    st.warning("Chatbot initialization failed. Please check your data files and API key.")