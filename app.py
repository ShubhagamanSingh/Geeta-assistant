import os
import pickle
import streamlit as st  # type: ignore
import faiss  # type: ignore
import numpy as np  # type: ignore
import re
from huggingface_hub import InferenceClient  # type: ignore
from dotenv import load_dotenv  # type: ignore
import json
from datetime import datetime
import sqlite3
import hashlib

# --- Load environment variables ---
load_dotenv()

# --- Streamlit Page Configuration ---
st.set_page_config(
    page_title="Geeta Assistant", 
    page_icon="🕉️", 
    layout="centered",
    initial_sidebar_state="collapsed"
)

# --- Enhanced Custom Styling ---
st.markdown(
    """
<style>
    .stApp {
        background: linear-gradient(135deg, #1a1a1a 0%, #2d1b69 100%);
        color: white;
    }

    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 0 0 30px 30px;
        margin-bottom: 2rem;
        text-align: center;
        box-shadow: 0 8px 32px rgba(0,0,0,0.3);
    }

    .main-header h1 {
        color: white;
        font-size: 3rem;
        margin: 0;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }

    .main-header h3 {
        color: rgba(255,255,255,0.9);
        margin: 0.5rem 0 0 0;
        font-weight: 300;
    }

    .chat-container {
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
        border-radius: 20px;
        padding: 2rem;
        margin: 1rem 0;
        border: 1px solid rgba(255,255,255,0.2);
    }

    .st-chat-message-user {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        color: white !important;
        padding: 1rem 1.5rem;
        border-radius: 20px 20px 5px 20px;
        margin: 10px 0;
        border: none;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
    }

    .st-chat-message-assistant {
        background: rgba(255, 255, 255, 0.15) !important;
        color: white !important;
        padding: 1rem 1.5rem;
        border-radius: 20px 20px 20px 5px;
        margin: 10px 0;
        border: 1px solid rgba(255,255,255,0.1);
        backdrop-filter: blur(10px);
    }

    .stChatInput input {
        background: rgba(255, 255, 255, 0.1) !important;
        color: white !important;
        border: 1px solid rgba(255,255,255,0.3) !important;
        border-radius: 25px !important;
        padding: 15px 20px !important;
        font-size: 1rem !important;
    }

    .stChatInput input:focus {
        border-color: #667eea !important;
        box-shadow: 0 0 0 2px rgba(102, 126, 234, 0.2) !important;
    }

    .stChatInput input::placeholder {
        color: rgba(255,255,255,0.6) !important;
    }

    .feature-card {
        background: rgba(255, 255, 255, 0.1);
        padding: 1.5rem;
        border-radius: 15px;
        margin: 1rem 0;
        border: 1px solid rgba(255,255,255,0.2);
        transition: all 0.3s ease;
    }

    .feature-card:hover {
        transform: translateY(-5px);
        background: rgba(255, 255, 255, 0.15);
    }

    .verse-of-day {
        background: linear-gradient(135deg, #FF6B6B 0%, #4ECDC4 100%);
        padding: 2rem;
        border-radius: 20px;
        text-align: center;
        margin: 2rem 0;
        box-shadow: 0 8px 32px rgba(0,0,0,0.2);
    }

    .chapter-info {
        background: rgba(255, 255, 255, 0.05);
        padding: 1rem;
        border-radius: 15px;
        margin: 1rem 0;
        border-left: 4px solid #667eea;
    }

    /* Hide default Streamlit elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    .stDeployButton {display: none;}
</style>
""",
    unsafe_allow_html=True,
)

# --- Local SQLite Database for User Sessions ---
def init_database():
    """Initialize SQLite database for user sessions"""
    conn = sqlite3.connect('geeta_sessions.db')
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS user_sessions (
            session_id TEXT PRIMARY KEY,
            username TEXT,
            chat_history TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            last_active TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    conn.commit()
    conn.close()

def save_session(session_id, username, chat_history):
    """Save user session to database"""
    conn = sqlite3.connect('geeta_sessions.db')
    c = conn.cursor()
    chat_json = json.dumps(chat_history)
    c.execute('''
        INSERT OR REPLACE INTO user_sessions 
        (session_id, username, chat_history, last_active)
        VALUES (?, ?, ?, CURRENT_TIMESTAMP)
    ''', (session_id, username, chat_json))
    conn.commit()
    conn.close()

def load_session(session_id):
    """Load user session from database"""
    conn = sqlite3.connect('geeta_sessions.db')
    c = conn.cursor()
    c.execute('SELECT chat_history FROM user_sessions WHERE session_id = ?', (session_id,))
    result = c.fetchone()
    conn.close()
    if result:
        return json.loads(result[0])
    return None

# --- Text Cleaning Functions ---
def is_meaningful_text(text, min_meaningful_chars=10):
    """Check if text contains meaningful content"""
    if not text or len(text.strip()) < min_meaningful_chars:
        return False

    # Remove special characters and check if enough text remains
    clean_text = re.sub(r"[^a-zA-Z0-9\u0900-\u097F\s]", "", text)
    clean_text = clean_text.strip()

    # Check if it's mostly special characters or garbage
    if len(clean_text) < min_meaningful_chars:
        return False

    # Check for repetitive patterns (like ~~~ ^~ ^~)
    repetitive_pattern = re.search(r"(.)\1{3,}", text)
    if repetitive_pattern:
        return False

    return True

def clean_chunk_text(text):
    """Clean and extract meaningful parts from text chunk"""
    lines = text.split("\n")
    meaningful_lines = []

    for line in lines:
        clean_line = line.strip()
        if is_meaningful_text(clean_line):
            meaningful_lines.append(clean_line)

    return " ".join(meaningful_lines[:3])  # Return first 3 meaningful lines

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

        st.info(f"📖 Loaded {len(chunks)} chunks from Bhagavad Gita")

        # Clean the chunks
        cleaned_chunks = []
        for i, chunk in enumerate(chunks):
            cleaned = clean_chunk_text(str(chunk))
            if cleaned and len(cleaned) > 20:  # Only keep meaningful chunks
                cleaned_chunks.append(cleaned)

        st.info(f"✨ After cleaning: {len(cleaned_chunks)} meaningful chunks remaining")

        if len(cleaned_chunks) == 0:
            st.error("❌ No meaningful text found in chunks. Please check your data file.")
            return None, None

        st.info("🔧 Creating FAISS index in memory. Please wait...")

        # Process embeddings in smaller batches
        batch_size = 5
        all_embeddings = []

        progress_bar = st.progress(0)
        for i in range(0, len(cleaned_chunks), batch_size):
            batch = cleaned_chunks[i : i + batch_size]
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

# --- Enhanced Emotional Support Responses ---
def get_emotional_support_response(query, context):
    """
    Provide specialized emotional support responses for difficult situations
    """
    query_lower = query.lower()

    # Enhanced emotional support patterns with more context
    emotional_keywords = {
        "downfall": {
            "response": "O dear one, even Arjuna felt despair on the battlefield. Remember Krishna's words: 'No one who does good work will ever come to a bad end, either here or in the world to come.' (Bhagavad Gita 6.40)",
            "verse": "Bhagavad Gita 6.40"
        },
        "sad": {
            "response": "The Gita teaches that happiness and distress come and go like seasons. 'The contacts of the senses with the sense objects give rise to heat and cold, pleasure and pain. They come and go, they are transient. Endure them bravely, O Arjuna.' (Bhagavad Gita 2.14)",
            "verse": "Bhagavad Gita 2.14"
        },
        "depress": {
            "response": "When depression clouds your vision, remember: 'The soul is eternal, indestructible, infinite. It is not slain when the body is slain. Therefore, fight, O Arjuna!' (Bhagavad Gita 2.18-19) Your true self is beyond temporary emotions.",
            "verse": "Bhagavad Gita 2.18-19"
        },
        "hope": {
            "response": "Have hope, dear seeker. Krishna promises: 'To those who are constantly devoted and worship Me with love, I give the understanding by which they can come to Me.' (Bhagavad Gita 10.10) The divine grace is always available.",
            "verse": "Bhagavad Gita 10.10"
        },
        "failure": {
            "response": "What you call failure is but a bend in the river of life. The Gita says: 'You have a right to perform your prescribed duty, but you are not entitled to the fruits of action.' (Bhagavad Gita 2.47) Perform your duty without attachment to success or failure.",
            "verse": "Bhagavad Gita 2.47"
        },
        "stress": {
            "response": "In times of stress, practice equanimity. 'A person is said to be established in self-realization and is called a yogi when he is fully satisfied by virtue of acquired knowledge and realization.' (Bhagavad Gita 6.8) Find peace within.",
            "verse": "Bhagavad Gita 6.8"
        },
        "anxiety": {
            "response": "For anxiety, the Gita prescribes: 'Therefore, without being attached to the fruits of activities, one should act as a matter of duty, for by working without attachment one attains the Supreme.' (Bhagavad Gita 3.19) Let go of expectations.",
            "verse": "Bhagavad Gita 3.19"
        },
        "lost": {
            "response": "When you feel lost, remember: 'Whenever there is a decline in righteousness and an increase in unrighteousness, O Arjuna, at that time I manifest Myself on earth.' (Bhagavad Gita 4.7) The divine is always with you, guiding your path.",
            "verse": "Bhagavad Gita 4.7"
        },
        "confus": {
            "response": "In confusion, seek clarity through knowledge. 'When your mind has overcome the confusion of duality, you will attain the state of perfect yoga.' (Bhagavad Gita 2.48) Steady your mind through meditation.",
            "verse": "Bhagavad Gita 2.48"
        },
        "trouble": {
            "response": "In troubled times, remember: 'The soul can never be cut to pieces by any weapon, nor burned by fire, nor moistened by water, nor withered by the wind.' (Bhagavad Gita 2.23) Your true self is eternal and indestructible.",
            "verse": "Bhagavad Gita 2.23"
        },
        "fear": {
            "response": "Cast away fear, O brave one. Krishna says: 'Give up all varieties of religiousness and just surrender unto Me. I shall deliver you from all sinful reactions. Do not fear.' (Bhagavad Gita 18.66)",
            "verse": "Bhagavad Gita 18.66"
        },
        "anger": {
            "response": "From anger comes delusion; from delusion, loss of memory; from loss of memory, the destruction of intelligence; from destruction of intelligence, one perishes. (Bhagavad Gita 2.63) Control your anger through wisdom.",
            "verse": "Bhagavad Gita 2.63"
        }
    }

    # Check for emotional keywords
    for keyword, data in emotional_keywords.items():
        if keyword in query_lower:
            return f"🕉️ **{data['verse']}**\n\n{data['response']}"

    # General emotional support with context
    return f"""🕉️ **Divine Guidance**

O dear seeker, I feel your pain and understand your struggle. 

In the Bhagavad Gita, Lord Krishna teaches Arjuna who was also in deep despair:

*"Never was there a time when I did not exist, nor you, nor all these kings; nor in the future shall any of us cease to be."* (Bhagavad Gita 2.12)

This too shall pass, my friend. The soul is eternal, and your current difficulties are but temporary circumstances. 

Have faith, perform your duty without attachment to results, and remember that every experience - both pleasant and painful - serves your spiritual growth.

**What specific aspect of your situation troubles you most?**"""

# --- Verse of the Day ---
def get_verse_of_day():
    """Get a random verse from Bhagavad Gita for the day"""
    verses = [
        {
            "verse": "You have the right to work, but never to the fruit of work.",
            "chapter": "2",
            "number": "47"
        },
        {
            "verse": "When meditation is mastered, the mind is unwavering like the flame of a candle in a windless place.",
            "chapter": "6",
            "number": "19"
        },
        {
            "verse": "Set thy heart upon thy work, but never on its reward.",
            "chapter": "2",
            "number": "47"
        },
        {
            "verse": "The soul is neither born, and nor does it die.",
            "chapter": "2",
            "number": "20"
        },
        {
            "verse": "There is neither this world, nor the world beyond. Nor happiness for the one who doubts.",
            "chapter": "4",
            "number": "40"
        }
    ]
    
    # Use day of year to get consistent daily verse
    day_of_year = datetime.now().timetuple().tm_yday
    selected_verse = verses[day_of_year % len(verses)]
    
    return selected_verse

# --- Enhanced RAG Logic ---
def get_geeta_response(query, chunks, index, k=5):
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
            context = "\n\n".join(meaningful_contexts[:3])
        else:
            context = "The Bhagavad Gita teaches eternal truths about duty, devotion, and the nature of reality."

        # Enhanced emotional detection
        emotional_words = [
            "downfall", "sad", "depress", "hope", "failure", 
            "stress", "anxiety", "lost", "confus", "trouble",
            "fear", "anger", "worried", "scared", "hopeless"
        ]
        
        if any(word in query.lower() for word in emotional_words):
            return get_emotional_support_response(query, context)

        # Try to use Hugging Face model for non-emotional queries
        try:
            client = InferenceClient(token=os.getenv("HUGGINGFACEHUB_API_TOKEN"))

            enhanced_prompt = f"""As Lord Krishna, provide divine guidance from the Bhagavad Gita with compassion and wisdom.

Context from Bhagavad Gita:
{context[:800]}

Seeker's Question: {query}

Respond as Krishna with:
- Direct reference to Gita teachings
- Practical spiritual advice  
- Compassionate tone
- Clear verse references when possible
- Encouragement for the spiritual path

Krishna's Answer:"""

            response = client.text_generation(
                enhanced_prompt, 
                model="Qwen/Qwen2.5-7B-Instruct", 
                max_new_tokens=300, 
                temperature=0.7
            )

            if response and len(response.strip()) > 30:
                # Add Krishna signature
                if "Krishna" not in response and "Gita" not in response:
                    response += f"\n\n🕉️ *This is the eternal wisdom of the Bhagavad Gita.*"
                return response

        except Exception as e:
            st.warning("✨ AI model contemplating deeply...")

        # Enhanced fallback response
        return f"""🕉️ **Divine Wisdom from Bhagavad Gita**

O seeker, your question touches upon eternal truths. While specific verses may vary, the core teachings remain:

• **Perform your duty** without attachment to results (2.47)
• **Cultivate equanimity** in success and failure (2.48)  
• **Offer all actions** to the divine (3.30)
• **Seek self-realization** through knowledge and devotion

Your current situation, though challenging, serves your spiritual evolution. Trust in the divine plan and continue walking the path of righteousness with steadfast devotion.

**What specific aspect of the Gita's wisdom would you like to explore deeper?**"""

    except Exception as e:
        st.error(f"❌ Retrieval failed: {e}")
        return get_emotional_support_response(query, "")

# --- Features Display ---
def display_features():
    """Display feature cards"""
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="feature-card">
            <h3>🎯 Personalized Guidance</h3>
            <p>Get specific advice from Bhagavad Gita tailored to your situation</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="feature-card">
            <h3>💖 Emotional Support</h3>
            <p>Find comfort and wisdom during difficult times</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="feature-card">
            <h3>📚 Verse References</h3>
            <p>Direct references to authentic Gita verses</p>
        </div>
        """, unsafe_allow_html=True)

# --- Chapter Information ---
def display_chapter_info():
    """Display Bhagavad Gita chapter information"""
    chapters = {
        "1": "Arjuna's Dilemma - Observing the Armies",
        "2": "The Eternal Reality - Soul & Karma Yoga",
        "3": "Path of Selfless Service - Karma Yoga",
        "4": "Path of Knowledge - Jnana Yoga",
        "5": "Path of Renunciation",
        "6": "Path of Meditation - Dhyana Yoga",
        "7": "Knowledge of the Absolute",
        "8": "The Supreme Goal",
        "9": "The Royal Knowledge",
        "10": "The Divine Glories",
        "11": "The Universal Form",
        "12": "Path of Devotion - Bhakti Yoga",
        "13": "The Field and Knower of the Field",
        "14": "The Three Modes of Nature",
        "15": "The Supreme Self",
        "16": "The Divine and Demonic Natures",
        "17": "The Threefold Faith",
        "18": "The Perfection of Renunciation"
    }
    
    st.markdown("### 📖 Bhagavad Gita Chapters")
    selected_chapter = st.selectbox("Select a chapter to learn about:", list(chapters.keys()), format_func=lambda x: f"Chapter {x}: {chapters[x]}")
    
    if selected_chapter:
        st.markdown(f"""
        <div class="chapter-info">
            <h4>Chapter {selected_chapter}: {chapters[selected_chapter]}</h4>
            <p>The Bhagavad Gita consists of 18 chapters with 700 verses, delivering profound spiritual wisdom through the dialogue between Lord Krishna and Arjuna.</p>
        </div>
        """, unsafe_allow_html=True)

# --- Main App Interface ---
def main():
    # Initialize database
    init_database()
    
    # Enhanced Header
    st.markdown("""
    <div class="main-header">
        <h1>🕉️ Geeta Assistant</h1>
        <h3>Your Personal Guide to Bhagavad Gita Wisdom</h3>
    </div>
    """, unsafe_allow_html=True)

    # Verse of the Day
    verse_data = get_verse_of_day()
    st.markdown(f"""
    <div class="verse-of-day">
        <h3>📜 Verse of the Day</h3>
        <p style="font-size: 1.2rem; font-style: italic; margin: 1rem 0;">"{verse_data['verse']}"</p>
        <p style="margin: 0; opacity: 0.9;">Bhagavad Gita Chapter {verse_data['chapter']}, Verse {verse_data['number']}</p>
    </div>
    """, unsafe_allow_html=True)

    # Display Features
    display_features()

    # Ensure token exists
    if not os.getenv("HUGGINGFACEHUB_API_TOKEN"):
        st.error("⚠️ Missing Hugging Face API key in your .env file.")
        st.info("Get a free token from: https://huggingface.co/settings/tokens")
        st.stop()

    # Load chatbot components
    chunks, index = load_chatbot_components()

    if all([chunks is not None, index is not None]):
        # Initialize session state
        if "messages" not in st.session_state:
            st.session_state.messages = [
                {
                    "role": "assistant",
                    "content": "🕉️ **Welcome, O Seeker!**\n\nI am Krishna, your guide to the eternal wisdom of the Bhagavad Gita. Whether you seek guidance on life's challenges, spiritual understanding, or emotional support, I am here to illuminate your path with divine knowledge.\n\n**What brings you to seek the Gita's wisdom today?**",
                }
            ]
        
        # Generate session ID
        if "session_id" not in st.session_state:
            st.session_state.session_id = hashlib.md5(str(datetime.now()).encode()).hexdigest()

        # Chat container
        st.markdown('<div class="chat-container">', unsafe_allow_html=True)
        st.markdown("### 💬 Divine Dialogue")

        # Display chat messages
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        # Chat input
        if prompt := st.chat_input("Ask about life, duty, spirituality..."):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                with st.spinner("🕉️ Krishna is contemplating your question..."):
                    answer = get_geeta_response(prompt, chunks, index)
                    st.markdown(answer)
                    st.session_state.messages.append({"role": "assistant", "content": answer})
            
            # Save session
            save_session(
                st.session_state.session_id, 
                "user", 
                st.session_state.messages
            )

        st.markdown('</div>', unsafe_allow_html=True)

        # Chapter Information
        display_chapter_info()

        # Quick Action Buttons
        st.markdown("### 🎯 Quick Guidance")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("💖 Emotional Support", use_container_width=True):
                st.session_state.messages.append({"role": "user", "content": "I need emotional support and comfort"})
                with st.chat_message("user"):
                    st.markdown("I need emotional support and comfort")
                with st.chat_message("assistant"):
                    answer = get_emotional_support_response("emotional support", "")
                    st.markdown(answer)
                    st.session_state.messages.append({"role": "assistant", "content": answer})
        
        with col2:
            if st.button("🎯 Life Purpose", use_container_width=True):
                st.session_state.messages.append({"role": "user", "content": "What does Gita say about finding life's purpose?"})
                with st.chat_message("user"):
                    st.markdown("What does Gita say about finding life's purpose?")
                with st.chat_message("assistant"):
                    answer = get_geeta_response("life purpose", chunks, index)
                    st.markdown(answer)
                    st.session_state.messages.append({"role": "assistant", "content": answer})
        
        with col3:
            if st.button("☮️ Inner Peace", use_container_width=True):
                st.session_state.messages.append({"role": "user", "content": "How to find inner peace according to Bhagavad Gita?"})
                with st.chat_message("user"):
                    st.markdown("How to find inner peace according to Bhagavad Gita?")
                with st.chat_message("assistant"):
                    answer = get_geeta_response("inner peace", chunks, index)
                    st.markdown(answer)
                    st.session_state.messages.append({"role": "assistant", "content": answer})

        # Clear chat button
        if st.button("🗑️ Clear Conversation", use_container_width=True):
            st.session_state.messages = [
                {
                    "role": "assistant",
                    "content": "🕉️ **Conversation Cleared**\n\nThe mind is now fresh like a clean slate. What wisdom shall we explore today, O seeker?",
                }
            ]
            st.rerun()

    else:
        st.warning("❌ Chatbot initialization failed. Please check your data files and API key.")

    # Footer
    st.markdown("---")
    st.markdown(
        "<div style='text-align: center; color: rgba(255,255,255,0.6);'>"
        "🕉️ May the wisdom of Bhagavad Gita illuminate your path 🕉️<br>"
        "Built with devotion using Streamlit & AI"
        "</div>",
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()