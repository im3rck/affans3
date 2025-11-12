"""Agentic AI System with CrewAI and ReAct Reasoning"""
import os
from typing import List, Optional
from dotenv import load_dotenv
from crewai import Agent, Task, Crew, Process
from langchain_google_genai import ChatGoogleGenerativeAI
from custom_tools import (
    KnowledgeBaseSearchTool,
    ProductQueryTool,
    WebSearchTool,
    QueryRewriterTool
)
from rag_system import HybridSearchRAG


class AgenticChatbot:
    """Agentic chatbot with ReAct reasoning using CrewAI"""

    def __init__(
        self,
        rag_system: HybridSearchRAG,
        api_key: Optional[str] = None,
        model_name: str = "gemini-1.5-pro",
        temperature: float = 0.3
    ):
        load_dotenv()

        self.api_key = api_key or os.getenv("GOOGLE_API_KEY")
        if not self.api_key:
            raise ValueError("Google API key is required. Set GOOGLE_API_KEY in .env file")

        self.model_name = model_name
        self.rag_system = rag_system

        # Initialize Gemini LLM
        self.llm = ChatGoogleGenerativeAI(
            model=model_name,
            google_api_key=self.api_key,
            temperature=temperature,
            convert_system_message_to_human=True
        )

        # Initialize custom tools
        self.knowledge_search_tool = KnowledgeBaseSearchTool(rag_system=rag_system)
        self.product_query_tool = ProductQueryTool()
        self.web_search_tool = WebSearchTool()
        self.query_rewriter_tool = QueryRewriterTool()

        # Create agents
        self._create_agents()

    def _create_agents(self):
        """Create specialized agents for different tasks"""

        # Main Research Agent with ReAct reasoning
        self.research_agent = Agent(
            role="Knowledge Research Specialist",
            goal=(
                "Find accurate and relevant information to answer user queries by "
                "intelligently searching the knowledge base, product database, and web resources."
            ),
            backstory=(
                "You are an expert information retrieval specialist at Amazon. "
                "You excel at understanding user needs, breaking down complex queries, "
                "and finding the most relevant information from various sources. "
                "You use a systematic ReAct (Reasoning + Acting) approach:\n"
                "1. REASON: Analyze the query and plan your search strategy\n"
                "2. ACT: Execute searches using appropriate tools\n"
                "3. OBSERVE: Review the results and assess relevance\n"
                "4. REPEAT: Refine your approach if needed\n"
                "You always cite your sources and provide accurate, helpful information."
            ),
            tools=[
                self.knowledge_search_tool,
                self.product_query_tool,
                self.query_rewriter_tool,
                self.web_search_tool
            ],
            llm=self.llm,
            verbose=True,
            allow_delegation=False,
            max_iter=10
        )

        # Answer Synthesis Agent
        self.synthesis_agent = Agent(
            role="Answer Synthesis Expert",
            goal=(
                "Synthesize information from multiple sources into clear, accurate, "
                "and helpful responses for users."
            ),
            backstory=(
                "You are an expert communicator who specializes in taking complex "
                "information from multiple sources and creating clear, concise, and "
                "helpful responses. You always maintain a friendly, professional tone "
                "and ensure that answers are accurate and well-structured."
            ),
            tools=[],
            llm=self.llm,
            verbose=True,
            allow_delegation=False
        )

    def create_research_task(self, query: str) -> Task:
        """Create a research task for the agent"""
        return Task(
            description=(
                f"User Query: {query}\n\n"
                "Your task:\n"
                "1. ANALYZE the query to understand what information is needed\n"
                "2. PLAN your search strategy:\n"
                "   - Is this about company policies/procedures? → Use Knowledge Base Search\n"
                "   - Is this about products? → Use Product Query\n"
                "   - Is this unclear or needs refinement? → Use Query Rewriter first\n"
                "   - Does it need external/real-time info? → Use Web Search\n"
                "3. EXECUTE your search plan using the appropriate tools\n"
                "4. EVALUATE the results - are they relevant and sufficient?\n"
                "5. If needed, REFINE your search with different queries or tools\n"
                "6. COMPILE all relevant information found\n\n"
                "Return a comprehensive summary of all information found, including sources."
            ),
            agent=self.research_agent,
            expected_output=(
                "A detailed compilation of relevant information with sources cited. "
                "Include specific details like procedures, steps, prices, ratings, etc."
            )
        )

    def create_synthesis_task(self, query: str) -> Task:
        """Create a synthesis task to generate the final answer"""
        return Task(
            description=(
                f"User Query: {query}\n\n"
                "Using the research findings from the previous task:\n"
                "1. Synthesize the information into a clear, helpful response\n"
                "2. Structure the answer logically (use bullet points, steps, etc.)\n"
                "3. Include specific details (prices, procedures, contact info, etc.)\n"
                "4. Cite sources when providing factual information\n"
                "5. If information is incomplete, acknowledge what's missing\n"
                "6. Maintain a friendly, professional tone\n\n"
                "Create a response that directly addresses the user's question."
            ),
            agent=self.synthesis_agent,
            expected_output=(
                "A clear, well-structured answer that directly addresses the user's query "
                "with specific details and source citations."
            )
        )

    def process_query(self, query: str) -> str:
        """
        Process a user query using the agentic system

        This implements the ReAct (Reasoning + Acting) pattern:
        - The research agent reasons about what tools to use
        - Executes actions using those tools
        - Observes the results
        - Iterates until sufficient information is gathered
        - The synthesis agent creates the final response
        """

        # Create tasks
        research_task = self.create_research_task(query)
        synthesis_task = self.create_synthesis_task(query)

        # Create crew with sequential process
        crew = Crew(
            agents=[self.research_agent, self.synthesis_agent],
            tasks=[research_task, synthesis_task],
            process=Process.sequential,
            verbose=True
        )

        # Execute
        try:
            result = crew.kickoff()
            return str(result)
        except Exception as e:
            return f"Error processing query: {str(e)}\nPlease try rephrasing your question."

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
            "Amazon_Customer_Service_Operations_KB.pdf",
            "Amazon_HR_Employee_Support_KB.pdf",
            "Amazon_IT_Device_Support_KB.pdf"
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
        "How do I track my order?",
        "What are the best USB cables under 500 rupees?",
        "What is the leave policy for sick days?"
    ]

    for query in test_queries:
        result = chatbot.chat(query)
        print(f"\n{'='*60}")
        print(f"RESPONSE:\n{result['response']}")
        print(f"{'='*60}\n")
