"""
SearXNG Web Search client.
"""

import requests
from typing import List, Dict, Optional

SEARXNG_URL = "http://localhost:8888"


def web_search(query: str, num_results: int = 5) -> List[Dict]:
    """
    Perform a web search using SearXNG.
    
    Args:
        query: Search query
        num_results: Number of results to return
        
    Returns:
        List of search results with title, url, and content
    """
    params = {
        "q": query,
        "format": "json",
        "categories": "general",
    }
    
    try:
        response = requests.get(
            f"{SEARXNG_URL}/search",
            params=params,
            timeout=30
        )
        response.raise_for_status()
        result = response.json()
        
        results = result.get("results", [])[:num_results]
        return [
            {
                "title": r.get("title", ""),
                "url": r.get("url", ""),
                "content": r.get("content", "")
            }
            for r in results
        ]
    except requests.exceptions.RequestException as e:
        raise RuntimeError(f"SearXNG request failed: {e}")


def format_search_context(results: List[Dict]) -> str:
    """
    Format search results into context string for LLM.
    
    Args:
        results: List of search results
        
    Returns:
        Formatted context string
    """
    if not results:
        return ""
    
    context_parts = []
    for i, r in enumerate(results, 1):
        context_parts.append(f"[{i}] {r['title']}\n{r['content']}\nSource: {r['url']}")
    
    return "\n\n".join(context_parts)


def search_and_format(query: str, num_results: int = 5) -> str:
    """
    Perform search and return formatted context.
    
    Args:
        query: Search query
        num_results: Number of results
        
    Returns:
        Formatted context string
    """
    results = web_search(query, num_results)
    return format_search_context(results)


if __name__ == "__main__":
    # Test the client
    results = web_search("capital of France")
    print("Search Results:")
    for r in results:
        print(f"- {r['title']}: {r['content'][:100]}...")
    
    print("\nFormatted Context:")
    print(search_and_format("capital of France"))
