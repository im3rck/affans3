# 🤖 RAG-based Chatbot with Agentic AI

An advanced AI chatbot system that combines **Retrieval-Augmented Generation (RAG)** with **Agentic AI** to provide intelligent, context-aware responses. Built with Google Gemini Pro, CrewAI, and advanced retrieval techniques.

## 🌟 Key Features

### Advanced RAG Implementation
- ✅ **Hybrid Search**: Combines vector similarity (semantic) and BM25 (keyword) search
- ✅ **Reranking**: Uses FlashRank for improved result relevance
- ✅ **Query Rewriting**: Automatically improves queries for better retrieval
- ✅ **Multi-source Knowledge**: Integrates PDFs, CSV data, and web search

### Agentic AI with ReAct
- 🧠 **Autonomous Decision Making**: Agent reasons about which tools to use
- 🔄 **ReAct Loop**: Reasoning → Acting → Observing → Iterating
- 🛠️ **Custom Tools**: 4 specialized tools for different tasks
- 🤝 **Multi-Agent System**: Research agent + Synthesis agent collaboration

### Technical Stack
- **LLM**: Google Gemini 1.5 Pro
- **Agent Framework**: CrewAI
- **Vector DB**: ChromaDB
- **Embeddings**: Sentence Transformers (all-MiniLM-L6-v2)
- **Reranking**: FlashRank (ms-marco-MiniLM-L-12-v2)
- **UI**: Streamlit

## 📁 Project Structure

```
affans3/
├── src/
│   ├── __init__.py
│   ├── document_processor.py    # PDF & CSV processing
│   ├── rag_system.py            # Hybrid search & reranking
│   ├── custom_tools.py          # Agent tools
│   └── agent_system.py          # CrewAI agents with ReAct
├── data/
│   ├── Amazon_Customer_Service_Operations_KB.pdf
│   ├── Amazon_HR_Employee_Support_KB.pdf
│   ├── Amazon_IT_Device_Support_KB.pdf
│   └── amazon.csv
├── app.py                        # Streamlit web interface
├── cli.py                        # Command-line interface
├── config.yaml                   # Configuration
├── requirements.txt              # Dependencies
└── .env.example                  # Environment variables template
```

## 🚀 Quick Start

### Prerequisites
- Python 3.8 or higher
- Google Gemini API key ([Get one here](https://makersuite.google.com/app/apikey))

### Installation

1. **Clone the repository**
```bash
git clone <repository-url>
cd affans3
```

2. **Create virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Configure environment**
```bash
cp .env.example .env
# Edit .env and add your GOOGLE_API_KEY
```

5. **Run the application**

**Streamlit Web Interface:**
```bash
streamlit run app.py
```

**Command Line Interface:**
```bash
python cli.py
```

## ⚙️ Configuration

### Environment Variables (.env)
```bash
GOOGLE_API_KEY=your_gemini_api_key_here
GEMINI_MODEL=gemini-1.5-pro
EMBEDDING_MODEL=models/embedding-001
CHUNK_SIZE=1000
CHUNK_OVERLAP=200
TOP_K_RESULTS=5
RERANK_TOP_K=3
CHROMA_PERSIST_DIRECTORY=./chroma_db
```

### Configuration File (config.yaml)
```yaml
rag:
  chunk_size: 1000
  chunk_overlap: 200
  top_k_results: 5
  rerank_top_k: 3
  hybrid_search:
    vector_weight: 0.7    # Weight for semantic search
    bm25_weight: 0.3      # Weight for keyword search

agent:
  max_iterations: 10
  verbose: true
  allow_delegation: false
```

## 🛠️ System Architecture

### 1. Document Processing Pipeline
```
PDFs + CSV → Text Extraction → Chunking → Embeddings → Vector DB
```

### 2. Hybrid Retrieval System
```
Query → [Vector Search + BM25 Search] → Hybrid Fusion → Reranking → Top-K Results
```

### 3. Agentic AI Flow (ReAct Pattern)
```
User Query → Research Agent (Reasoning) → Tool Selection (Acting)
→ Tool Execution → Result Observation → Iteration (if needed)
→ Synthesis Agent → Final Response
```

## 🔧 Custom Tools

### 1. Knowledge Base Search Tool
- Searches internal knowledge base (PDFs)
- Uses hybrid search with reranking
- Domains: Customer Service, HR, IT Support

### 2. Product Query Tool
- Searches product database (CSV)
- Filters by category, rating, price
- Returns top products with reviews

### 3. Query Rewriter Tool
- Expands abbreviations (VPN → virtual private network)
- Generates query variations
- Improves search effectiveness

### 4. Web Search Tool
- External information retrieval
- Real-time data (configurable)
- Falls back when knowledge base insufficient

## 📊 Advanced RAG Techniques

### Hybrid Search
Combines two complementary approaches:
- **Vector Search (70%)**: Semantic similarity using embeddings
- **BM25 Search (30%)**: Keyword matching with TF-IDF

### Reranking
- First retrieval: Get top 5 candidates from hybrid search
- Reranking: Use cross-encoder to reorder by relevance
- Final selection: Return top 3 most relevant documents

### Query Rewriting
- Automatic expansion of abbreviations
- Alternative phrasings generation
- Context-aware query enhancement

## 🤖 Agent Architecture

### Research Agent
- **Role**: Knowledge Research Specialist
- **Strategy**: ReAct (Reasoning + Acting)
- **Tools**: All 4 custom tools
- **Goal**: Find accurate, relevant information

### Synthesis Agent
- **Role**: Answer Synthesis Expert
- **Strategy**: Information consolidation
- **Tools**: None (uses research results)
- **Goal**: Create clear, helpful responses

### ReAct Loop Implementation
```python
1. REASON: Analyze query and plan strategy
   - Is this a policy question? → Knowledge Base
   - Is this about products? → Product Query
   - Need clarification? → Query Rewriter

2. ACT: Execute searches with selected tools
   - Call appropriate tool(s)
   - Gather information

3. OBSERVE: Review results
   - Assess relevance and completeness
   - Determine if more information needed

4. ITERATE: Refine if needed
   - Try different queries
   - Use alternative tools
   - Maximum 10 iterations

5. SYNTHESIZE: Create final response
```

## 📝 Usage Examples

### Example 1: Policy Question
```
User: "What is the leave policy for sick days?"

Agent Process:
1. Reasoning: This is about HR policy → Use Knowledge Base Search
2. Acting: Search HR knowledge base
3. Observing: Found relevant policy information
4. Synthesizing: Create structured response with policy details
```

### Example 2: Product Query
```
User: "Find the best USB cables under 500 rupees"

Agent Process:
1. Reasoning: Product search with price filter → Use Product Query
2. Acting: Query products (category=USB, price<500, sort by rating)
3. Observing: Retrieved top 5 products
4. Synthesizing: Format with prices, ratings, reviews
```

### Example 3: Complex Multi-step Query
```
User: "How do I setup VPN and what if I have issues?"

Agent Process:
1. Reasoning: IT support question, may need query rewriting
2. Acting: Rewrite "setup VPN" → "VPN setup procedure configuration"
3. Acting: Search knowledge base with improved query
4. Observing: Found setup instructions
5. Reasoning: User also asks about troubleshooting
6. Acting: Search for "VPN troubleshooting issues"
7. Synthesizing: Combine setup + troubleshooting in response
```

## 🎯 Knowledge Domains

### Customer Service Operations
- Order tracking procedures
- Refund request handling
- Returns management
- Escalation matrix
- Complaint handling
- Performance metrics

### HR & Employee Support
- Leave policies (sick, vacation, PTO)
- Payroll information
- Employee benefits
- Performance reviews
- Promotion cycles
- WFH guidelines

### IT & Device Support
- Laptop setup procedures
- VPN configuration
- ServiceNow ticketing
- MFA setup
- Password resets
- Software requests

### Product Information
- 1,465 products in database
- Product specifications
- Pricing and discounts
- Customer ratings
- User reviews

## 🧪 Testing

### Test Individual Components
```bash
# Test document processor
cd src
python document_processor.py

# Test RAG system
python rag_system.py

# Test agent system
python agent_system.py
```

### Test Full System
```bash
# Web interface
streamlit run app.py

# CLI
python cli.py
```

## 📈 Performance Optimization

### Indexing
- First run: ~2-3 minutes (builds vector index)
- Subsequent runs: <10 seconds (loads existing index)
- Total chunks: ~1,500 documents

### Query Performance
- Hybrid search: ~0.5-1 second
- Reranking: ~0.2-0.3 seconds
- Total retrieval: ~1-2 seconds
- Agent reasoning: ~5-10 seconds (depends on complexity)

## 🔒 Security & Privacy

- API keys stored in `.env` (gitignored)
- No data sent to external services except Gemini API
- Local vector database (ChromaDB)
- Configurable data retention

## 🤝 Contributing

Contributions welcome! Areas for improvement:
- [ ] Add more advanced RAG techniques (HyDE, Self-RAG)
- [ ] Implement conversation memory
- [ ] Add multi-modal support (images)
- [ ] Integrate real web search API
- [ ] Add evaluation metrics (RAGAS)

## 📄 License

This project is for educational and demonstration purposes.

## 🙏 Acknowledgments

- **Google Gemini**: LLM provider
- **CrewAI**: Agent framework
- **LangChain**: Document processing utilities
- **ChromaDB**: Vector database
- **FlashRank**: Reranking model

## 📞 Support

For issues or questions:
1. Check existing issues in the repository
2. Create a new issue with detailed description
3. Include error logs and configuration (remove API keys!)

## 🎓 Learning Resources

- [RAG Fundamentals](https://python.langchain.com/docs/use_cases/question_answering/)
- [CrewAI Documentation](https://docs.crewai.com/)
- [Google Gemini API](https://ai.google.dev/docs)
- [ReAct Pattern](https://arxiv.org/abs/2210.03629)

---

**Built with ❤️ using Google Gemini Pro, CrewAI, and Advanced RAG Techniques**
