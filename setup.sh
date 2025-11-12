#!/bin/bash

echo "================================================"
echo "  RAG Chatbot with Agentic AI - Setup Script"
echo "================================================"
echo ""

# Check Python version
echo "Checking Python version..."
python_version=$(python3 --version 2>&1)
echo "Found: $python_version"

if ! command -v python3 &> /dev/null; then
    echo "❌ Error: Python 3 is not installed"
    exit 1
fi

# Create virtual environment
echo ""
echo "Creating virtual environment..."
python3 -m venv venv

# Activate virtual environment
echo "Activating virtual environment..."
if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "win32" ]]; then
    source venv/Scripts/activate
else
    source venv/bin/activate
fi

# Upgrade pip
echo ""
echo "Upgrading pip..."
pip install --upgrade pip

# Install requirements
echo ""
echo "Installing dependencies..."
pip install -r requirements.txt

# Create .env file if it doesn't exist
if [ ! -f .env ]; then
    echo ""
    echo "Creating .env file from template..."
    cp .env.example .env
    echo "✅ Created .env file"
    echo ""
    echo "⚠️  IMPORTANT: Please edit .env and add your GOOGLE_API_KEY"
    echo "   Get your API key from: https://makersuite.google.com/app/apikey"
else
    echo ""
    echo "✅ .env file already exists"
fi

# Create directories
echo ""
echo "Creating necessary directories..."
mkdir -p chroma_db
mkdir -p logs

echo ""
echo "================================================"
echo "  ✅ Setup Complete!"
echo "================================================"
echo ""
echo "Next steps:"
echo "1. Edit .env file and add your GOOGLE_API_KEY"
echo "2. Activate the virtual environment:"
echo "   source venv/bin/activate  (Linux/Mac)"
echo "   venv\\Scripts\\activate     (Windows)"
echo "3. Run the application:"
echo "   streamlit run app.py      (Web UI)"
echo "   python cli.py             (Command Line)"
echo ""
echo "For more information, see README.md"
echo ""
