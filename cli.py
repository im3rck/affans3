"""
RAG-based Chatbot with Agentic AI
Command Line Interface
"""
import os
import sys
from dotenv import load_dotenv
import yaml

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.document_processor import DocumentProcessor
from src.rag_system import HybridSearchRAG
from src.agent_system import AgenticChatbot


def print_banner():
    """Print application banner"""
    print("\n" + "="*60)
    print("  🤖 Amazon AI Assistant - RAG Chatbot with Agentic AI")
    print("  Powered by Gemini Pro & CrewAI")
    print("="*60 + "\n")


def load_config():
    """Load configuration"""
    load_dotenv()
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    return config


def initialize_systems():
    """Initialize RAG and agent systems"""
    print("Initializing systems...\n")

    config = load_config()

    # Check API key
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        print("❌ Error: GOOGLE_API_KEY not found in .env file")
        print("Please create a .env file with your Gemini API key")
        sys.exit(1)

    # Initialize RAG
    persist_dir = os.getenv("CHROMA_PERSIST_DIRECTORY", "./chroma_db")

    if not os.path.exists(persist_dir):
        print("📚 Building knowledge base index (first time setup)...\n")

        # Process documents
        processor = DocumentProcessor(
            chunk_size=config['rag']['chunk_size'],
            chunk_overlap=config['rag']['chunk_overlap']
        )

        pdf_files = config['data_sources']['pdfs']
        csv_files = config['data_sources']['csv']

        documents = processor.process_all_documents(pdf_files, csv_files)

        # Initialize and index
        rag = HybridSearchRAG(
            persist_directory=persist_dir,
            vector_weight=config['rag']['hybrid_search']['vector_weight'],
            bm25_weight=config['rag']['hybrid_search']['bm25_weight']
        )
        rag.index_documents(documents)

        print(f"\n✅ Knowledge base indexed successfully!")
    else:
        print("📚 Loading existing knowledge base...\n")
        rag = HybridSearchRAG(
            persist_directory=persist_dir,
            vector_weight=config['rag']['hybrid_search']['vector_weight'],
            bm25_weight=config['rag']['hybrid_search']['bm25_weight']
        )
        print("✅ Knowledge base loaded!")

    # Initialize chatbot
    print("\n🤖 Initializing AI agent...\n")
    model_name = os.getenv("GEMINI_MODEL", "gemini-1.5-pro")
    chatbot = AgenticChatbot(
        rag_system=rag,
        model_name=model_name,
        temperature=0.3
    )
    print("✅ AI agent ready!\n")

    return chatbot


def print_help():
    """Print help message"""
    print("\nAvailable commands:")
    print("  /help    - Show this help message")
    print("  /clear   - Clear the screen")
    print("  /exit    - Exit the application")
    print("\nExample queries:")
    print("  - How do I track my order?")
    print("  - What is the leave policy for sick days?")
    print("  - Find USB cables under 500 rupees")
    print("  - How to setup VPN?")
    print()


def main():
    """Main CLI application"""
    print_banner()

    # Initialize
    try:
        chatbot = initialize_systems()
    except Exception as e:
        print(f"❌ Initialization error: {str(e)}")
        print("\nPlease make sure:")
        print("1. .env file exists with GOOGLE_API_KEY")
        print("2. PDF and CSV files are in the root directory")
        print("3. All dependencies are installed (pip install -r requirements.txt)")
        sys.exit(1)

    print_help()

    # Main loop
    print("="*60)
    print("Start chatting! (Type /help for commands, /exit to quit)")
    print("="*60 + "\n")

    while True:
        try:
            # Get user input
            user_input = input("You: ").strip()

            if not user_input:
                continue

            # Handle commands
            if user_input.startswith('/'):
                command = user_input.lower()

                if command == '/exit' or command == '/quit':
                    print("\n👋 Goodbye! Thanks for using Amazon AI Assistant!\n")
                    break

                elif command == '/help':
                    print_help()
                    continue

                elif command == '/clear':
                    os.system('clear' if os.name != 'nt' else 'cls')
                    print_banner()
                    continue

                else:
                    print(f"❌ Unknown command: {user_input}")
                    print("Type /help for available commands\n")
                    continue

            # Process query
            print("\n🤖 AI Assistant:")
            print("-" * 60)

            try:
                result = chatbot.chat(user_input)
                print(result['response'])
            except Exception as e:
                print(f"❌ Error: {str(e)}")
                print("Please try rephrasing your question.")

            print("-" * 60 + "\n")

        except KeyboardInterrupt:
            print("\n\n👋 Goodbye! Thanks for using Amazon AI Assistant!\n")
            break
        except Exception as e:
            print(f"\n❌ Unexpected error: {str(e)}\n")


if __name__ == "__main__":
    main()
