# 🚀 Quick Start Guide

Get your RAG-based Chatbot running in 5 minutes!

## Prerequisites

- Python 3.8 or higher
- Google Gemini API key

## Step-by-Step Setup

### 1️⃣ Get Your API Key

1. Go to [Google AI Studio](https://makersuite.google.com/app/apikey)
2. Sign in with your Google account
3. Click "Create API Key"
4. Copy your API key

### 2️⃣ Install Dependencies

**Option A: Using setup script (Linux/Mac)**
```bash
chmod +x setup.sh
./setup.sh
```

**Option B: Manual installation**
```bash
# Create virtual environment
python3 -m venv venv

# Activate it
source venv/bin/activate  # Linux/Mac
# OR
venv\Scripts\activate     # Windows

# Install packages
pip install -r requirements.txt

# Create .env file
cp .env.example .env
```

### 3️⃣ Configure API Key

Edit `.env` file:
```bash
GOOGLE_API_KEY=your_actual_api_key_here
```

### 4️⃣ Run the Application

**Web Interface (Recommended)**
```bash
streamlit run app.py
```
Then open http://localhost:8501 in your browser

**Command Line Interface**
```bash
python cli.py
```

## 🎯 Try These Queries

Once running, try asking:

### Knowledge Base Queries
```
How do I track my order?
What is the leave policy?
How to setup VPN?
How to escalate a customer complaint?
```

### Product Queries
```
Find USB cables under 500 rupees
Show me high-rated electronics
Best products with good reviews
```

### Complex Queries
```
I need a VPN setup guide and troubleshooting tips
What are the refund policies and how long does it take?
```

## 🔧 Troubleshooting

### "Module not found" error
```bash
# Make sure you activated the virtual environment
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows

# Reinstall requirements
pip install -r requirements.txt
```

### "API key not found" error
```bash
# Check .env file exists
ls -la .env

# Verify it contains your API key
cat .env

# Make sure GOOGLE_API_KEY is set correctly
```

### "ChromaDB error" or slow first run
- First run takes 2-3 minutes to build the index
- Subsequent runs will be much faster
- If issues persist, delete `chroma_db/` folder and restart

### "Gemini API error"
- Verify your API key is valid
- Check you have API quota remaining
- Ensure you're using Gemini 1.5 Pro (not deprecated models)

## 📊 What Happens on First Run?

1. **Document Processing** (30 seconds)
   - Reads 3 PDF files
   - Processes CSV with 1,465 products
   - Creates ~1,500 text chunks

2. **Index Building** (1-2 minutes)
   - Generates embeddings for all chunks
   - Builds vector database (ChromaDB)
   - Creates BM25 keyword index

3. **Agent Initialization** (10 seconds)
   - Loads Gemini Pro model
   - Initializes 4 custom tools
   - Sets up 2-agent system

**Total first run: ~3-4 minutes**

**Subsequent runs: ~10 seconds** (loads existing index)

## 💡 Tips

- Use specific queries for better results
- The agent will automatically choose the right tools
- Check the "Response Details" in Streamlit to see which tools were used
- First query after startup may be slower (model initialization)

## 🎓 Understanding the Response

The chatbot uses a multi-step process:

```
Your Query
    ↓
Research Agent (analyzes query)
    ↓
Tool Selection (picks best tools)
    ↓
Information Retrieval
    ↓
Synthesis Agent (creates answer)
    ↓
Final Response
```

You'll see this process in action in verbose mode!

## 📚 Next Steps

- Read the full [README.md](README.md) for detailed documentation
- Explore [config.yaml](config.yaml) for customization options
- Check [src/](src/) folder to understand the code
- Modify tools in `src/custom_tools.py` for your use case

## 🆘 Need Help?

- Check the [README.md](README.md) for detailed docs
- Review error messages carefully
- Ensure all prerequisites are met
- Create an issue if you find a bug

---

**Happy chatting! 🤖**
