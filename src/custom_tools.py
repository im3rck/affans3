"""Custom Tools for the Agentic AI System"""
from typing import Optional, Type, Any
from pydantic import BaseModel, Field
from crewai_tools import BaseTool
from rag_system import HybridSearchRAG
import pandas as pd
import requests
from datetime import datetime


class KnowledgeBaseSearchInput(BaseModel):
    """Input schema for KnowledgeBaseSearchTool"""
    query: str = Field(..., description="The search query to find relevant information in the knowledge base")


class KnowledgeBaseSearchTool(BaseTool):
    """Tool for searching the knowledge base using advanced RAG"""
    name: str = "Knowledge Base Search"
    description: str = (
        "Search the Amazon knowledge base for information about:\n"
        "- HR & Benefits (pay stubs, PTO, 401k, leave of absence, W-2, holidays, employment verification)\n"
        "- IT & Tech Support (VPN, password reset, BitLocker, peripheral devices, mobile setup, phishing, ServiceNow)\n"
        "- Workplace & Operations (conference rooms, office supplies, building maintenance, visitor registration, safety, access badges)\n"
        "Use this tool when you need to answer questions about company policies, procedures, or support."
    )
    args_schema: Type[BaseModel] = KnowledgeBaseSearchInput
    rag_system: Optional[HybridSearchRAG] = None

    def __init__(self, rag_system: HybridSearchRAG):
        super().__init__()
        self.rag_system = rag_system

    def _run(self, query: str) -> str:
        """Execute the knowledge base search"""
        try:
            # Retrieve relevant documents
            documents = self.rag_system.retrieve(query, top_k=5, rerank_top_k=3)

            if not documents:
                return "No relevant information found in the knowledge base."

            # Format results
            result_parts = ["Retrieved Information:\n"]
            for i, doc in enumerate(documents, 1):
                source = doc.metadata.get('filename', doc.metadata.get('source', 'Unknown'))
                page = doc.metadata.get('page', '')
                page_info = f" (Page {page})" if page else ""

                result_parts.append(f"\n--- Source {i}: {source}{page_info} ---")
                result_parts.append(doc.page_content)
                result_parts.append("-" * 50)

            return "\n".join(result_parts)

        except Exception as e:
            return f"Error searching knowledge base: {str(e)}"


class ProductQueryInput(BaseModel):
    """Input schema for ProductQueryTool"""
    query: str = Field(
        ...,
        description="Natural language query about products (e.g., 'USB cables under 500 rupees with good ratings')"
    )
    filter_category: Optional[str] = Field(
        None,
        description="Filter by product category (optional)"
    )
    min_rating: Optional[float] = Field(
        None,
        description="Minimum rating filter (0-5)"
    )


class ProductQueryTool(BaseTool):
    """Tool for querying product information from the CSV database"""
    name: str = "Product Query"
    description: str = (
        "Query the Amazon product database to find products based on:\n"
        "- Product name, category, or description\n"
        "- Price range and discounts\n"
        "- Ratings and reviews\n"
        "Use this tool when users ask about specific products, prices, or product recommendations."
    )
    args_schema: Type[BaseModel] = ProductQueryInput
    csv_file: str = "amazon.csv"
    df: Optional[pd.DataFrame] = None

    def __init__(self):
        super().__init__()
        try:
            self.df = pd.read_csv(self.csv_file)
        except Exception as e:
            print(f"Error loading CSV: {e}")
            self.df = None

    def _run(
        self,
        query: str,
        filter_category: Optional[str] = None,
        min_rating: Optional[float] = None
    ) -> str:
        """Execute the product query"""
        try:
            if self.df is None:
                return "Product database is not available."

            # Start with full dataframe
            filtered_df = self.df.copy()

            # Apply filters
            if filter_category:
                filtered_df = filtered_df[
                    filtered_df['category'].str.contains(filter_category, case=False, na=False)
                ]

            if min_rating:
                filtered_df = filtered_df[filtered_df['rating'] >= min_rating]

            # Search in product name and description
            query_lower = query.lower()
            mask = (
                filtered_df['product_name'].str.lower().str.contains(query_lower, na=False) |
                filtered_df['about_product'].str.lower().str.contains(query_lower, na=False) |
                filtered_df['category'].str.lower().str.contains(query_lower, na=False)
            )
            results_df = filtered_df[mask]

            # Sort by rating and rating count
            results_df = results_df.sort_values(
                by=['rating', 'rating_count'],
                ascending=[False, False]
            ).head(5)

            if results_df.empty:
                return f"No products found matching the query: {query}"

            # Format results
            result_parts = [f"Found {len(results_df)} products:\n"]

            for idx, row in results_df.iterrows():
                result_parts.append(f"\n{'='*60}")
                result_parts.append(f"Product: {row['product_name']}")
                result_parts.append(f"Category: {row['category']}")
                result_parts.append(f"Price: {row['discounted_price']} (Original: {row['actual_price']})")
                result_parts.append(f"Discount: {row['discount_percentage']}")
                result_parts.append(f"Rating: {row['rating']}/5 ({row['rating_count']} ratings)")

                if pd.notna(row['about_product']):
                    description = str(row['about_product'])[:200]
                    result_parts.append(f"Description: {description}...")

                if pd.notna(row['review_content']):
                    review = str(row['review_content'])[:150]
                    result_parts.append(f"Sample Review: {review}...")

            return "\n".join(result_parts)

        except Exception as e:
            return f"Error querying products: {str(e)}"


class WebSearchInput(BaseModel):
    """Input schema for WebSearchTool"""
    query: str = Field(..., description="The search query for real-time web information")


class WebSearchTool(BaseTool):
    """Tool for searching the web for real-time information"""
    name: str = "Web Search"
    description: str = (
        "Search the web for real-time information about:\n"
        "- Current events and news\n"
        "- Latest product information\n"
        "- Updated policies or procedures\n"
        "- External resources and documentation\n"
        "Use this tool when the query requires information not available in the knowledge base."
    )
    args_schema: Type[BaseModel] = WebSearchInput

    def _run(self, query: str) -> str:
        """Execute the web search"""
        try:
            # Note: This is a simulated web search
            # In production, integrate with Google Custom Search API, Serper API, etc.

            return (
                f"Web Search Results for: '{query}'\n\n"
                f"[Simulated Web Search - In production, this would use a real search API]\n\n"
                f"To enable real web search:\n"
                f"1. Sign up for Google Custom Search API or Serper API\n"
                f"2. Add API key to .env file\n"
                f"3. Implement actual search logic\n\n"
                f"For now, please refer to the knowledge base or product database for information."
            )

        except Exception as e:
            return f"Error performing web search: {str(e)}"


class QueryRewriterInput(BaseModel):
    """Input schema for QueryRewriterTool"""
    original_query: str = Field(..., description="The original user query to rewrite")
    context: Optional[str] = Field(None, description="Additional context for rewriting")


class QueryRewriterTool(BaseTool):
    """Tool for rewriting queries to improve search results"""
    name: str = "Query Rewriter"
    description: str = (
        "Rewrite user queries to improve search effectiveness by:\n"
        "- Expanding abbreviations and acronyms\n"
        "- Adding relevant keywords\n"
        "- Reformulating ambiguous queries\n"
        "- Breaking down complex multi-part questions\n"
        "Use this tool when the initial search returns poor results."
    )
    args_schema: Type[BaseModel] = QueryRewriterInput

    def _run(self, original_query: str, context: Optional[str] = None) -> str:
        """Rewrite the query for better search results"""
        try:
            # Simple query rewriting logic
            rewritten_queries = []

            # Original query
            rewritten_queries.append(original_query)

            # Expand common abbreviations
            expansions = {
                'vpn': 'virtual private network',
                'hr': 'human resources',
                'it': 'information technology',
                'mfa': 'multi-factor authentication',
                'wfh': 'work from home',
                'pto': 'paid time off'
            }

            expanded_query = original_query.lower()
            for abbr, full in expansions.items():
                if abbr in expanded_query:
                    expanded_query = expanded_query.replace(abbr, f"{abbr} {full}")
                    rewritten_queries.append(expanded_query)

            # Add question variations
            if 'how' in original_query.lower():
                rewritten_queries.append(original_query.replace('how', 'steps to'))
                rewritten_queries.append(original_query.replace('how', 'procedure for'))

            # Remove duplicates while preserving order
            unique_queries = []
            for q in rewritten_queries:
                if q not in unique_queries:
                    unique_queries.append(q)

            result = "Rewritten Queries:\n"
            for i, q in enumerate(unique_queries, 1):
                result += f"{i}. {q}\n"

            return result

        except Exception as e:
            return f"Error rewriting query: {str(e)}"


if __name__ == "__main__":
    # Test the tools
    print("Testing Product Query Tool...")
    product_tool = ProductQueryTool()
    result = product_tool._run("USB cable", min_rating=4.0)
    print(result)

    print("\n" + "="*60 + "\n")
    print("Testing Query Rewriter Tool...")
    rewriter_tool = QueryRewriterTool()
    result = rewriter_tool._run("how to setup vpn?")
    print(result)
