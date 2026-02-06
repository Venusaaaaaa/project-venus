import streamlit as st
import base64
from agent import create_agent

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Car Sensor Assistant",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CUSTOM CSS STYLING ---
st.markdown("""
<style>
    /* Global Styles */
    .stApp {
        background: linear-gradient(to bottom right, #f8f9fa, #e9ecef);
        font-family: 'Segoe UI', sans-serif;
    }
    
    /* Chat Message Styling */
    .stChatMessage {
        background-color: white;
        border-radius: 12px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        margin-bottom: 10px;
        border: 1px solid #dee2e6;
    }
    
    /* Header Styling */
    h1 {
        color: #2c3e50;
        font-weight: 700;
    }
    
    h2, h3 {
        color: #34495e;
    }
    
    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: #2c3e50;
        color: white;
    }
    
    section[data-testid="stSidebar"] h1, 
    section[data-testid="stSidebar"] h2, 
    section[data-testid="stSidebar"] h3,
    section[data-testid="stSidebar"] span,
    section[data-testid="stSidebar"] p {
        color: white !important;
    }

    /* Presentation Cards */
    .project-card {
        background-color: white;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin-bottom: 20px;
    }
</style>
""", unsafe_allow_html=True)

# --- INITIALIZE AGENT ---
if "agent" not in st.session_state:
    st.session_state.agent = create_agent()

# --- INITIALIZE CHAT HISTORY ---
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hello! I'm your Car Sensor Assistant. Ask me anything about MAF, O2, or other sensors!"}
    ]

# --- SIDEBAR NAVIGATION ---
with st.sidebar:
    st.title("🚗 Menu")
    
    page = st.radio(
        "Navigate to:", 
        ["Chat Assistant", "Project Presentation"]
    )
    
    st.markdown("---")
    st.markdown("### 🛠 Tools")
    if st.button("Clear Chat History", type="primary"):
        st.session_state.messages = [
            {"role": "assistant", "content": "Chat history cleared. How can I help you?"}
        ]
        st.rerun()
    
    st.markdown("---")
    st.info("Powered by LangGraph & Google Gemini")


# --- MAIN CONTENT ---

if page == "Chat Assistant":
    st.title("💬 Car Sensor Assistant")
    st.caption("AI-powered diagnostics for your vehicle sensors.")

    # Display chat messages from history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Accept user input
    if prompt := st.chat_input("Describe your car problem..."):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display user message in chat message container
        with st.chat_message("user"):
            st.markdown(prompt)

        # Display assistant response in chat message container
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            message_placeholder.markdown("⏳ *Thinking...*")
            
            try:
                # Call the agent
                result = st.session_state.agent.invoke({"input": prompt})
                final_answer = result["final_answer"]
                
                # Update placeholder with actual response
                message_placeholder.markdown(final_answer)
                
                # Add assistant response to chat history
                st.session_state.messages.append({"role": "assistant", "content": final_answer})
            
            except Exception as e:
                error_msg = f"❌ An error occurred: {str(e)}"
                message_placeholder.error(error_msg)
                st.session_state.messages.append({"role": "assistant", "content": error_msg})

elif page == "Project Presentation":
    st.title("📽 Project Overview: Car Sensor Assistant")
    
    st.markdown("""
    <div class="project-card">
        <h3>🚀 Project Goal</h3>
        <p>
            To create an intelligent assistant that helps mechanics and car owners diagnose sensor-related issues 
            using Retrieval-Augmented Generation (RAG).
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="project-card">
            <h3>🏗 Architecture</h3>
            <ul>
                <li><strong>LangChain & LangGraph</strong>: Manages the reasoning flow.</li>
                <li><strong>Google Gemini</strong>: Provides natural language understanding.</li>
                <li><strong>FAISS</strong>: Vector database for fast information retrieval.</li>
                <li><strong>Streamlit</strong>: The interactive user interface you see here.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
    with col2:
        st.markdown("""
        <div class="project-card">
            <h3>✨ Key Features</h3>
            <ul>
                <li><strong>Context Aware</strong>: Remembers conversation history.</li>
                <li><strong>RAG Powered</strong>: Uses a specific car sensor manual (PDF) for accuracy.</li>
                <li><strong>Interactive Chat</strong>: Easy-to-use troubleshooting interface.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("### 🧠 How Retrieval-Augmented Generation (RAG) Works")
    st.markdown("""
    1.  **Ingestion**: The Sensor Guide PDF is split into chunks and embedded into vectors.
    2.  **Retrieval**: When you ask a question, the system finds the most relevant chunks.
    3.  **Generation**: The AI combines the retrieved info with your question to generate a precise answer.
    """)
    
    st.success("This project demonstrates a practical application of GenAI for technical support!")
