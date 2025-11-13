"""Agentic AI System with CrewAI and ReAct Reasoning"""
import os
import time
from typing import List, Optional
from dotenv import load_dotenv
from crewai import Agent, Task, Crew, Process, LLM
from custom_tools import initialize_tools, get_all_tools
from rag_system import HybridSearchRAG


class AgenticChatbot:
    """Agentic chatbot with ReAct reasoning using CrewAI"""

    def __init__(
        self,
        rag_system: HybridSearchRAG,
        api_key: Optional[str] = None,
        model_name: str = "gemini/gemini-1.5-flash-latest",
        temperature: float = 0.3
    ):
        load_dotenv()

        self.api_key = api_key or os.getenv("GOOGLE_API_KEY")
        if not self.api_key:
            raise ValueError("Google API key is required. Set GOOGLE_API_KEY in .env file")

        # Set API key in environment for LiteLLM
        os.environ["GEMINI_API_KEY"] = self.api_key

        self.model_name = model_name
        self.rag_system = rag_system

        # Initialize Gemini LLM using CrewAI's LLM wrapper
        self.llm = LLM(
            model=model_name,
            temperature=temperature
        )

        # Initialize custom tools
        initialize_tools(rag_system, "amazon.csv")
        self.tools = get_all_tools()

        # Create agents
        self._create_agents()

    def _create_agents(self):
        """Create specialized agents for different tasks"""

        # Single agent that both researches and synthesizes
        self.assistant_agent = Agent(
            role="AI Assistant",
            goal=(
                "Answer user queries accurately by searching the knowledge base and "
                "providing clear, helpful responses."
            ),
            backstory=(
                "You are an AI assistant at Amazon. You help employees and customers find "
                "information by searching the knowledge base, product database, and other resources. "
                "You provide clear, concise answers with proper source citations."
            ),
            tools=self.tools,
            llm=self.llm,
            verbose=True,
            allow_delegation=False,
            max_iter=5  # Reduced to save API calls
        )

    def create_task(self, query: str) -> Task:
        """Create a task for the agent"""
        return Task(
            description=(
                f"Answer this question: {query}\n\n"
                "Instructions:\n"
                "1. Use Knowledge Base Search for HR, IT, or workplace questions\n"
                "2. Use Product Query for product-related questions\n"
                "3. Provide a clear, concise answer with sources\n"
                "4. Keep your response focused and helpful"
            ),
            agent=self.assistant_agent,
            expected_output=(
                "A clear answer to the user's question with source citations."
            )
        )

    def process_query(self, query: str, max_retries: int = 3) -> str:
        """
        Process a user query using the agentic system with retry logic

        This implements the ReAct (Reasoning + Acting) pattern with rate limit handling.
        """

        # Create task
        task = self.create_task(query)

        # Create crew with single agent
        crew = Crew(
            agents=[self.assistant_agent],
            tasks=[task],
            process=Process.sequential,
            verbose=True
        )

        # Execute with retry logic for rate limits
        for attempt in range(max_retries):
            try:
                result = crew.kickoff()
                return str(result)
            except Exception as e:
                error_msg = str(e)

                # Check if it's a rate limit error
                if "429" in error_msg or "RESOURCE_EXHAUSTED" in error_msg or "quota" in error_msg.lower():
                    if attempt < max_retries - 1:
                        # Calculate wait time (exponential backoff: 2, 4, 8 seconds)
                        wait_time = 2 ** (attempt + 1)
                        print(f"⏳ Rate limit hit. Waiting {wait_time} seconds before retry {attempt + 2}/{max_retries}...")
                        time.sleep(wait_time)
                        continue
                    else:
                        return (
                            "❌ Rate limit exceeded. The free tier allows 15 requests/minute for Gemini Flash.\n\n"
                            "Solutions:\n"
                            "1. Wait 1-2 minutes and try again\n"
                            "2. Upgrade to a paid Gemini API plan for higher limits\n"
                            "3. Use simpler queries that require fewer tool calls"
                        )
                else:
                    return f"Error processing query: {error_msg}\n\nPlease try rephrasing your question."

        return "Failed after multiple retries. Please try again later."

    def chat(self, query: str) -> dict:
        """
        Main chat interface

        Returns:
            dict with 'query', 'response', and 'metadata'
        """
        print(f"\n{'='*60}")
        print(f"User Query: {query}")
        print(f"{'='*60}\n")

        response = self.process_query(query)

        return {
            "query": query,
            "response": response,
            "metadata": {
                "model": self.model_name,
                "tools_used": [
                    "Knowledge Base Search",
                    "Product Query",
                    "Query Rewriter",
                    "Web Search"
                ]
            }
        }


if __name__ == "__main__":
    # Test the agent system
    from document_processor import DocumentProcessor

    print("Initializing RAG system...")

    # Check if index exists
    if not os.path.exists("./chroma_db"):
        print("Building index...")
        processor = DocumentProcessor()
        pdf_files = [
            "HR & Benefits FAQ.pdf",
            "IT & Tech Support.pdf",
            "Workplace & Operations FAQ.pdf"
        ]
        csv_files = ["amazon.csv"]
        documents = processor.process_all_documents(pdf_files, csv_files)

        rag = HybridSearchRAG()
        rag.index_documents(documents)
    else:
        print("Loading existing index...")
        rag = HybridSearchRAG()

    print("\nInitializing agent system...")
    chatbot = AgenticChatbot(rag_system=rag)

    # Test queries
    test_queries = [
        "How do I view my pay stub?",
        "What are the best USB cables under 500 rupees?",
        "How do I reset my network password?"
    ]

    for query in test_queries:
        result = chatbot.chat(query)
        print(f"\n{'='*60}")
        print(f"RESPONSE:\n{result['response']}")
        print(f"{'='*60}\n")
