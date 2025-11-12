# 🏗️ System Architecture

Detailed technical documentation of the RAG-based Chatbot with Agentic AI.

## 📋 Table of Contents

1. [System Overview](#system-overview)
2. [RAG Pipeline](#rag-pipeline)
3. [Agentic AI System](#agentic-ai-system)
4. [Data Flow](#data-flow)
5. [Component Details](#component-details)

## System Overview

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                         User Interface                       │
│                   (Streamlit / CLI)                          │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│                    Agentic AI Layer                          │
│  ┌──────────────────┐         ┌──────────────────┐          │
│  │ Research Agent   │────────▶│ Synthesis Agent  │          │
│  │  (ReAct Loop)    │         │                  │          │
│  └────────┬─────────┘         └──────────────────┘          │
│           │                                                  │
│           ▼                                                  │
│  ┌────────────────────────────────────────────┐             │
│  │           Custom Tools Layer               │             │
│  ├────────────┬──────────┬──────────┬────────┤             │
│  │  KB Search │ Product  │  Query   │  Web   │             │
│  │    Tool    │  Query   │ Rewriter │ Search │             │
│  └────┬───────┴────┬─────┴────┬─────┴───┬────┘             │
└───────┼────────────┼──────────┼─────────┼──────────────────┘
        │            │          │         │
        ▼            ▼          ▼         ▼
┌─────────────────────────────────────────────────────────────┐
│                      RAG Pipeline                            │
│  ┌──────────────┐  ┌───────────┐  ┌──────────────┐         │
│  │ Vector Search│  │    BM25   │  │  Reranking   │         │
│  │  (Semantic)  │  │ (Keyword) │  │ (FlashRank)  │         │
│  └──────┬───────┘  └─────┬─────┘  └──────┬───────┘         │
└─────────┼────────────────┼────────────────┼─────────────────┘
          │                │                │
          ▼                ▼                ▼
┌─────────────────────────────────────────────────────────────┐
│                    Data Layer                                │
│  ┌──────────────┐  ┌───────────┐  ┌──────────────┐         │
│  │   ChromaDB   │  │  BM25     │  │  Product DB  │         │
│  │ (Embeddings) │  │  Index    │  │   (CSV)      │         │
│  └──────────────┘  └───────────┘  └──────────────┘         │
└─────────────────────────────────────────────────────────────┘
```

## RAG Pipeline

### 1. Document Processing

```python
Document Processing Flow:
┌─────────────┐
│   PDFs +    │
│     CSV     │
└──────┬──────┘
       │
       ▼
┌────────────────────┐
│  Text Extraction   │
│  - PyMuPDF (PDFs)  │
│  - Pandas (CSV)    │
└──────┬─────────────┘
       │
       ▼
┌────────────────────────────┐
│    Text Chunking           │
│  Strategy: Recursive       │
│  Size: 1000 chars          │
│  Overlap: 200 chars        │
│  Separators: \n\n, \n, .   │
└──────┬─────────────────────┘
       │
       ▼
┌────────────────────────────┐
│    ~1,500 Chunks           │
│  Metadata: source, page,   │
│  type, category            │
└────────────────────────────┘
```

**Key Implementation:**
```python
# src/document_processor.py
class DocumentProcessor:
    def __init__(self, chunk_size=1000, chunk_overlap=200):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
```

### 2. Hybrid Search System

```python
Hybrid Search Flow:
┌─────────────┐
│  User Query │
└──────┬──────┘
       │
       ├─────────────┬────────────┐
       │             │            │
       ▼             ▼            ▼
┌─────────────┐ ┌─────────┐ ┌─────────┐
│   Vector    │ │  BM25   │ │ Query   │
│   Search    │ │ Search  │ │Expansion│
│  (0.7 wt)   │ │(0.3 wt) │ │         │
└──────┬──────┘ └────┬────┘ └─────────┘
       │             │
       ▼             ▼
┌──────────────────────────┐
│   Score Normalization    │
│   + Weighted Fusion      │
└──────┬───────────────────┘
       │
       ▼
┌──────────────────┐
│   Top-K=5        │
│   Candidates     │
└──────┬───────────┘
       │
       ▼
┌──────────────────┐
│   Reranking      │
│  (Cross-encoder) │
└──────┬───────────┘
       │
       ▼
┌──────────────────┐
│   Top-K=3        │
│   Final Results  │
└──────────────────┘
```

**Why Hybrid Search?**

1. **Vector Search (70% weight)**
   - Captures semantic meaning
   - Good for: conceptual queries, synonyms
   - Example: "track order" matches "shipment tracking"

2. **BM25 Search (30% weight)**
   - Keyword matching with TF-IDF
   - Good for: specific terms, codes, names
   - Example: "VPN" matches exact term

3. **Combined Strength**
   - Covers both semantic and lexical matching
   - Improves recall and precision

**Implementation:**
```python
# src/rag_system.py
class HybridSearchRAG:
    def hybrid_search(self, query, top_k=5):
        # Get results from both
        vector_results = self._vector_search(query, top_k*2)
        bm25_results = self._bm25_search(query, top_k*2)

        # Normalize scores
        vector_scores = self._normalize_scores(vector_results)
        bm25_scores = self._normalize_scores(bm25_results)

        # Weighted fusion
        combined_scores = (
            self.vector_weight * vector_scores +
            self.bm25_weight * bm25_scores
        )

        return top_k_documents
```

### 3. Reranking

**Purpose:** Cross-encoder reranking for higher precision

```python
Reranking Process:
┌─────────────────────┐
│ Initial Top-5       │
│ (from hybrid)       │
└──────┬──────────────┘
       │
       ▼
┌─────────────────────────────┐
│   FlashRank Cross-Encoder  │
│   Model: ms-marco-MiniLM   │
│                             │
│   Computes relevance score  │
│   for (query, document)     │
│   pair using BERT-style     │
│   cross-attention           │
└──────┬──────────────────────┘
       │
       ▼
┌─────────────────────┐
│ Reordered by        │
│ relevance score     │
└──────┬──────────────┘
       │
       ▼
┌─────────────────────┐
│ Final Top-3         │
│ (highest precision) │
└─────────────────────┘
```

**Benefits:**
- Improves precision by 15-30%
- Corrects ranking errors from initial retrieval
- Considers full query-document interaction

## Agentic AI System

### ReAct Pattern Implementation

```python
ReAct Loop:
┌──────────────────────────────────────┐
│         THOUGHT (Reasoning)          │
│  "This is an HR policy question,     │
│   I should search the knowledge      │
│   base for leave policy info"        │
└────────────┬─────────────────────────┘
             │
             ▼
┌──────────────────────────────────────┐
│          ACTION (Acting)             │
│  Tool: Knowledge Base Search         │
│  Query: "leave policy sick days"     │
└────────────┬─────────────────────────┘
             │
             ▼
┌──────────────────────────────────────┐
│        OBSERVATION (Observing)       │
│  Retrieved: "Sick leave policy...    │
│  Employees are entitled to 12        │
│  sick days per year..."              │
└────────────┬─────────────────────────┘
             │
             ▼
┌──────────────────────────────────────┐
│         THOUGHT (Reasoning)          │
│  "This information is sufficient.    │
│   I have the policy details."        │
└────────────┬─────────────────────────┘
             │
             ▼
┌──────────────────────────────────────┐
│     FINAL ANSWER (Synthesis)         │
│  "According to our leave policy,     │
│   employees are entitled to..."      │
└──────────────────────────────────────┘
```

### Multi-Agent System

**Agent 1: Research Agent**
```python
Role: Knowledge Research Specialist
Goal: Find relevant information
Tools:
  - Knowledge Base Search (PDFs)
  - Product Query (CSV)
  - Query Rewriter (optimization)
  - Web Search (external)

Process:
  1. Analyze query intent
  2. Plan search strategy
  3. Execute tool calls
  4. Evaluate results
  5. Iterate if needed (max 10)
  6. Compile findings
```

**Agent 2: Synthesis Agent**
```python
Role: Answer Synthesis Expert
Goal: Create clear responses
Tools: None (uses research results)

Process:
  1. Receive research findings
  2. Organize information
  3. Structure response
  4. Add citations
  5. Format for readability
```

### Tool Architecture

Each tool implements:
```python
class CustomTool(BaseTool):
    name: str              # Tool identifier
    description: str       # When to use this tool
    args_schema: BaseModel # Input validation

    def _run(self, **kwargs) -> str:
        # Tool logic
        # Returns: formatted string result
```

**Tool 1: Knowledge Base Search**
```python
Input: search query
Process:
  1. Generate query embedding
  2. Hybrid search (vector + BM25)
  3. Rerank top results
  4. Format with sources
Output: Retrieved documents with metadata
```

**Tool 2: Product Query**
```python
Input: query, category, min_rating
Process:
  1. Load product CSV
  2. Filter by criteria
  3. Text search in name/description
  4. Sort by rating
  5. Get top 5
Output: Formatted product list
```

**Tool 3: Query Rewriter**
```python
Input: original query
Process:
  1. Expand abbreviations
  2. Generate variations
  3. Add context
Output: List of rewritten queries
```

**Tool 4: Web Search**
```python
Input: search query
Process:
  1. [Simulated in current version]
  2. Can integrate Google Custom Search
  3. Or Serper API
Output: External information
```

## Data Flow

### Complete Query Processing Flow

```python
1. USER INPUT
   ↓
   "How do I track my order?"

2. STREAMLIT/CLI INTERFACE
   ↓
   Sends to AgenticChatbot.chat()

3. AGENTIC LAYER
   ↓
   Creates Research Task + Synthesis Task

4. RESEARCH AGENT (ReAct Loop)
   ↓
   THOUGHT: "This is about order tracking"
   ACTION: Use Knowledge Base Search Tool
   ↓

5. KNOWLEDGE BASE SEARCH TOOL
   ↓
   Calls rag_system.retrieve()

6. RAG SYSTEM
   ↓
   a) Vector Search (ChromaDB)
      - Embed query: [0.23, -0.45, ...]
      - Cosine similarity search
      - Top 10 candidates

   b) BM25 Search
      - Tokenize: ["track", "order"]
      - TF-IDF scoring
      - Top 10 candidates

   c) Hybrid Fusion
      - Normalize scores
      - Weighted combine (0.7 * vector + 0.3 * bm25)
      - Top 5 candidates

   d) Reranking
      - Cross-encoder scoring
      - Top 3 final results
   ↓

7. BACK TO TOOL
   ↓
   Format results with sources
   Return to Research Agent

8. RESEARCH AGENT
   ↓
   OBSERVATION: "Found tracking procedure"
   THOUGHT: "Sufficient information"
   Compile findings

9. SYNTHESIS AGENT
   ↓
   Receives research results
   Creates structured answer
   Adds formatting and citations

10. RESPONSE
    ↓
    Return to user interface

11. DISPLAY
    ↓
    Show formatted response with metadata
```

## Component Details

### ChromaDB Configuration

```python
Client: PersistentClient
Path: ./chroma_db
Distance: Cosine similarity
Embedding: all-MiniLM-L6-v2 (384 dims)
Index: HNSW (Hierarchical Navigable Small World)
```

**Why ChromaDB?**
- Fast approximate nearest neighbor search
- Persistent storage
- Easy to use
- Good for prototypes and production

### Embedding Model

```python
Model: all-MiniLM-L6-v2
Dimensions: 384
Speed: ~500 sentences/sec
Quality: Good for general purpose
Size: ~80MB
```

**Why this model?**
- Balance of speed and quality
- Multilingual support
- Well-tested in production
- Efficient for local deployment

### Reranking Model

```python
Model: ms-marco-MiniLM-L-12-v2
Type: Cross-encoder
Input: Query + Document pairs
Output: Relevance score
```

**Why FlashRank?**
- Faster than alternatives (4x)
- Same quality as larger models
- Optimized for CPU
- Low memory footprint

### LLM Configuration

```python
Model: Gemini 1.5 Pro
API: Google AI Studio
Temperature: 0.3 (for consistency)
Features:
  - 1M token context
  - Function calling support
  - Fast response time
  - Cost-effective
```

## Performance Characteristics

### Latency Breakdown

```python
Operation                Time
─────────────────────────────────
Document Processing      2-3 min (first time)
Index Loading           5-10 sec
Query Embedding         0.05 sec
Vector Search           0.3 sec
BM25 Search            0.2 sec
Reranking              0.3 sec
─────────────────────────────────
Total Retrieval         ~1 sec
─────────────────────────────────
Agent Reasoning         3-5 sec
LLM Generation         2-4 sec
─────────────────────────────────
Total Response          5-10 sec
```

### Scalability

```python
Current:
  - 1,500 documents
  - ~1 MB text data
  - Single user

Can scale to:
  - 100K+ documents
  - ~100 MB text data
  - Multiple concurrent users

Optimizations needed:
  - Batch processing
  - Query caching
  - Load balancing
  - GPU acceleration
```

## Advanced Features

### Query Rewriting Strategy

1. **Abbreviation Expansion**
   ```python
   "VPN issue" → "VPN virtual private network issue"
   ```

2. **Question Reformulation**
   ```python
   "How to X?" → "Steps to X"
   "How to X?" → "Procedure for X"
   ```

3. **Context Addition**
   ```python
   "Setup VPN" → "Setup VPN configuration guide"
   ```

### Context Window Management

```python
Strategy: Sliding Window
Max Context: 1000 tokens
Overlap: 200 tokens

Ensures:
  - No information loss at boundaries
  - Coherent chunks
  - Better semantic matching
```

### Error Handling

```python
Layer 1: Tool Level
  - Try-catch in each tool
  - Return error message as string
  - Agent can react to errors

Layer 2: Agent Level
  - Max iterations limit
  - Fallback strategies
  - Graceful degradation

Layer 3: Application Level
  - User-friendly error messages
  - Logging for debugging
  - Retry mechanisms
```

## Future Enhancements

### Potential Improvements

1. **Advanced RAG Techniques**
   - HyDE (Hypothetical Document Embeddings)
   - Self-RAG (self-reflective retrieval)
   - Multi-hop reasoning

2. **Better Memory**
   - Conversation history
   - User preferences
   - Session management

3. **Multi-Modal**
   - Image understanding
   - PDF table extraction
   - Chart analysis

4. **Evaluation**
   - RAGAS metrics
   - Human feedback loop
   - A/B testing

5. **Production Ready**
   - API endpoints
   - Authentication
   - Rate limiting
   - Monitoring

---

**This architecture provides a solid foundation for advanced RAG applications with agentic AI capabilities.**
