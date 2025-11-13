"""Custom Tools for the Agentic AI System"""
from typing import Optional
from crewai.tools import tool
from rag_system import HybridSearchRAG
import pandas as pd


# Global variables to store instances
_rag_system: Optional[HybridSearchRAG] = None
_product_df: Optional[pd.DataFrame] = None


def initialize_tools(rag_system: HybridSearchRAG, csv_file: str = "amazon.csv"):
    """Initialize the tools with required resources"""
    global _rag_system, _product_df
    _rag_system = rag_system
    try:
        _product_df = pd.read_csv(csv_file)
    except Exception as e:
        print(f"Warning: Could not load product CSV: {e}")
        _product_df = None


@tool("Knowledge Base Search")
def knowledge_base_search(query: str) -> str:
    """
    Search the Amazon knowledge base for information about:
    - HR & Benefits (pay stubs, PTO, 401k, leave of absence, W-2, holidays, employment verification)
    - IT & Tech Support (VPN, password reset, BitLocker, peripheral devices, mobile setup, phishing, ServiceNow)
    - Workplace & Operations (conference rooms, office supplies, building maintenance, visitor registration, safety, access badges)
    Use this tool when you need to answer questions about company policies, procedures, or support.

    Args:
        query: The search query to find relevant information in the knowledge base

    Returns:
        Retrieved information with sources
    """
    try:
        if _rag_system is None:
            return "Error: Knowledge base not initialized."

        # Retrieve relevant documents
        documents = _rag_system.retrieve(query, top_k=5, rerank_top_k=3)

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


@tool("Product Query")
def product_query(query: str, filter_category: Optional[str] = None, min_rating: Optional[float] = None) -> str:
    """
    Query the Amazon product database to find products based on:
    - Product name, category, or description
    - Price range and discounts
    - Ratings and reviews
    Use this tool when users ask about specific products, prices, or product recommendations.

    Args:
        query: Natural language query about products (e.g., 'USB cables under 500 rupees with good ratings')
        filter_category: Optional filter by product category
        min_rating: Optional minimum rating filter (0-5)

    Returns:
        Product information with prices, ratings, and reviews
    """
    try:
        if _product_df is None:
            return "Product database is not available."

        # Start with full dataframe
        filtered_df = _product_df.copy()

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


@tool("Query Rewriter")
def query_rewriter(original_query: str) -> str:
    """
    Rewrite user queries to improve search effectiveness by:
    - Expanding abbreviations and acronyms
    - Adding relevant keywords
    - Reformulating ambiguous queries
    - Breaking down complex multi-part questions
    Use this tool when the initial search returns poor results.

    Args:
        original_query: The original user query to rewrite

    Returns:
        List of rewritten query variations
    """
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
            'pto': 'paid time off',
            'faq': 'frequently asked questions',
            'kb': 'knowledge base'
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


@tool("Web Search")
def web_search(query: str) -> str:
    """
    Search the web for real-time information about:
    - Current events and news
    - Latest product information
    - Updated policies or procedures
    - External resources and documentation
    Use this tool when the query requires information not available in the knowledge base.

    Args:
        query: The search query for real-time web information

    Returns:
        Web search results
    """
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


# Function to get all tools
def get_all_tools():
    """Get all available tools as a list"""
    return [
        knowledge_base_search,
        product_query,
        query_rewriter,
        web_search
    ]


if __name__ == "__main__":
    # Test the tools
    print("Testing Product Query Tool...")
    initialize_tools(None, "amazon.csv")  # Initialize with None for RAG (won't test it)
    result = product_query("USB cable", min_rating=4.0)
    print(result)

    print("\n" + "="*60 + "\n")
    print("Testing Query Rewriter Tool...")
    result = query_rewriter("how to setup vpn?")
    print(result)
