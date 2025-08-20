import os
from typing import List, Dict, Any, Tuple
from dotenv import load_dotenv
from tavily import TavilyClient


# Load the env variables
load_dotenv()

class CorrectiveSearcher:
    """
    This class is a Corrective Searcher which aims:
        - To execute Web Searching as an extra tool for verification of RAG document scanning
        - If no documents are found, it will default to web search. 
    """
    def __init__(self):
        self.enabled = False

        # Try and get TAVILY Web Search API Key and setup Tavily Client
        try:
            api_key = os.getenv("TAVILY_API_KEY")
            self.client = TavilyClient(api_key=api_key)
            # Set enabled true by default
            if api_key:
                self.enabled = True
            else:
                print("DEBUG: No API key, web corrective disabled")
        except Exception as e:
            print(f"DEBUG: Error initializing Tavily client: {e}")
            self.client = None
            self.enabled = False

    # Main function to execute the web search
    def search(self, question: str, contexts: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[str]]:
        """
        Input: 
            - Query to search (str)
            - Contexts (list of contexts from documents RAG pipeline)
        
        Output:
            - Web Search Contexts (top 4)
            - Web Source Links
        """
        # If web search is not enabled then return empty
        if not self.enabled or not self.client:
            print("DEBUG: Web corrective not enabled or no client")
            return [], []
        
        # Only add context hint if we have meaningful context, but keep it very short
        if contexts and contexts[0].get("text", "").strip():
            # Take only first 100 characters and clean them up (since API limits)
            first_ctx = contexts[0]["text"][:100].strip()
            
            # Remove special characters and keep only alphanumeric + spaces
            first_ctx = " ".join(word for word in first_ctx.split() if word.isalnum() or word.isspace())[:80]
            if first_ctx:
                q = f"{question}. Focus on market size, competitors, pricing, regulation. Context: {first_ctx}"
            else:
                q = f"{question}. Focus on market size, competitors, pricing, regulation."
        
        # If no document contexts are available: clean query without context hint
        else:
            q = f"{question}. Focus on market size, competitors, pricing, regulation."
        
        # print(f"DEBUG: Search query: {q}")
        
        # Execute the client search function, default to empty lists if error
        try:
            res = self.client.search(q, search_depth="advanced", max_results=5)
            print(f"DEBUG: Response keys: {list(res.keys()) if res else 'None'}")
        except Exception as e:
            print(f"DEBUG: Tavily search error: {e}")
            return [], []
        
        # Initiate extra contexts and sources lists
        extra_contexts, source_urls = [], []
        results = res.get("results", [])

        # For each web searched result, we extract the URL and content from the results        
        for i, item in enumerate(results):
            print(f"DEBUG: Processing result {i}: {item}")
            url, content = item.get("url",""), item.get("content","")
            
            # If no content, we output print and continue
            if not content:
                print(f"DEBUG: Result {i} has no content, skipping")
                continue
            
            # Append extra content (max 1800 char) and append urls found
            extra_contexts.append({"text": content[:1800], "metadata": {"source": url, "chunk_index": 0}})
            source_urls.append(url)

        # Return extra contexts and web urls to the search        
        return extra_contexts[:4], source_urls
