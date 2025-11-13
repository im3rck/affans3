"""
RAG-based Chatbot with Agentic AI
Streamlit Application
"""
import os
import sys
import streamlit as st
from dotenv import load_dotenv
import yaml

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.document_processor import DocumentProcessor
from src.rag_system import HybridSearchRAG
from src.agent_system import AgenticChatbot


# Page configuration
st.set_page_config(
    page_title="Amazon AI Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #FF9900;
        text-align: center;
        padding: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        padding-bottom: 2rem;
    }
    .info-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .stChatMessage {
        padding: 1rem;
        border-radius: 0.5rem;
    }
    </style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_config():
    """Load configuration"""
    load_dotenv()
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    return config


@st.cache_resource
def initialize_rag_system(_config):
    """Initialize and cache the RAG system"""
    with st.spinner("🔄 Initializing RAG system..."):
        # Check if index exists
        persist_dir = os.getenv("CHROMA_PERSIST_DIRECTORY", "./chroma_db")

        if not os.path.exists(persist_dir):
            st.info("📚 Building knowledge base index (this may take a few minutes)...")

            # Process documents
            processor = DocumentProcessor(
                chunk_size=_config['rag']['chunk_size'],
                chunk_overlap=_config['rag']['chunk_overlap']
            )

            pdf_files = _config['data_sources']['pdfs']
            csv_files = _config['data_sources']['csv']

            documents = processor.process_all_documents(pdf_files, csv_files)

            # Initialize and index
            rag = HybridSearchRAG(
                persist_directory=persist_dir,
                vector_weight=_config['rag']['hybrid_search']['vector_weight'],
                bm25_weight=_config['rag']['hybrid_search']['bm25_weight']
            )
            rag.index_documents(documents)

            st.success(f"✅ Indexed {len(documents)} document chunks!")
        else:
            # Load existing index
            rag = HybridSearchRAG(
                persist_directory=persist_dir,
                vector_weight=_config['rag']['hybrid_search']['vector_weight'],
                bm25_weight=_config['rag']['hybrid_search']['bm25_weight']
            )
            st.success("✅ Loaded existing knowledge base!")

    return rag


@st.cache_resource
def initialize_chatbot(_rag_system, _config):
    """Initialize and cache the chatbot"""
    with st.spinner("🤖 Initializing AI agent..."):
        model_name = os.getenv("GEMINI_MODEL", "gemini-1.5-pro")
        chatbot = AgenticChatbot(
            rag_system=_rag_system,
            model_name=model_name,
            temperature=0.3
        )
        st.success("✅ AI agent ready!")
    return chatbot


def main():
    """Main application"""

    # Header
    st.markdown('<div class="main-header">🤖 Amazon AI Assistant</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-header">Powered by RAG, Agentic AI & Gemini Pro</div>',
        unsafe_allow_html=True
    )

    # Sidebar
    with st.sidebar:
        st.header("ℹ️ About")
        st.markdown("""
        This AI assistant helps you with:

        **📚 Knowledge Base:**
        - HR & Benefits (PTO, 401k, leave, payroll)
        - IT & Tech Support (VPN, passwords, devices)
        - Workplace & Operations (facilities, safety, access)

        **🛍️ Product Information:**
        - Product search & recommendations
        - Pricing & discounts
        - Reviews & ratings

        **🧠 Advanced Features:**
        - Hybrid Search (Vector + BM25)
        - Reranking for accuracy
        - Query rewriting
        - Agentic AI with ReAct reasoning
        """)

        st.divider()

        st.header("⚙️ Configuration")
        config = load_config()

        st.metric("Model", os.getenv("GEMINI_MODEL", "gemini-1.5-pro"))
        st.metric("Top K Results", config['rag']['top_k_results'])
        st.metric("Rerank Top K", config['rag']['rerank_top_k'])

        st.divider()

        st.header("📊 System Status")
        if 'rag_system' in st.session_state:
            st.success("✅ RAG System: Active")
        if 'chatbot' in st.session_state:
            st.success("✅ AI Agent: Active")

        st.divider()

        # Example queries
        st.header("💡 Example Queries")
        example_queries = [
            "How do I view my pay stub?",
            "How to reset my network password?",
            "How do I reserve a conference room?",
            "Find USB cables under ₹500",
            "What is the 401k enrollment process?",
            "How to report a phishing email?"
        ]

        for query in example_queries:
            if st.button(query, key=query, use_container_width=True):
                st.session_state.example_query = query

    # Main content
    try:
        # Load configuration
        config = load_config()

        # Initialize systems
        if 'rag_system' not in st.session_state:
            st.session_state.rag_system = initialize_rag_system(config)

        if 'chatbot' not in st.session_state:
            st.session_state.chatbot = initialize_chatbot(st.session_state.rag_system, config)

        # Initialize chat history
        if 'messages' not in st.session_state:
            st.session_state.messages = []

        # Display chat history
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        # Handle example query from sidebar
        if 'example_query' in st.session_state:
            query = st.session_state.example_query
            del st.session_state.example_query
        else:
            query = st.chat_input("Ask me anything about Amazon services or products...")

        # Process user input
        if query:
            # Display user message
            with st.chat_message("user"):
                st.markdown(query)

            # Add to history
            st.session_state.messages.append({"role": "user", "content": query})

            # Generate response
            with st.chat_message("assistant"):
                with st.spinner("🔍 Searching knowledge base and analyzing..."):
                    try:
                        result = st.session_state.chatbot.chat(query)
                        response = result['response']
                        st.markdown(response)

                        # Add metadata in expander
                        with st.expander("📋 Response Details"):
                            st.json({
                                "model": result['metadata']['model'],
                                "tools_available": result['metadata']['tools_used']
                            })

                    except Exception as e:
                        error_msg = f"❌ Error: {str(e)}\n\nPlease check your API key and try again."
                        st.error(error_msg)
                        response = error_msg

            # Add assistant response to history
            st.session_state.messages.append({"role": "assistant", "content": response})

        # Clear chat button
        if st.session_state.messages:
            col1, col2, col3 = st.columns([1, 1, 1])
            with col2:
                if st.button("🗑️ Clear Chat", use_container_width=True):
                    st.session_state.messages = []
                    st.rerun()

    except Exception as e:
        st.error(f"❌ Application Error: {str(e)}")
        st.info("💡 Make sure you have:\n1. Created a .env file with GOOGLE_API_KEY\n2. Placed the PDF and CSV files in the root directory")


if __name__ == "__main__":
    main()
