import os
from typing import List, Dict, Any, Tuple
from dotenv import load_dotenv

load_dotenv()

class CorrectiveSearcher:
    def __init__(self):
        self.enabled = False
        try:
            from tavily import TavilyClient
            api_key = os.getenv("TAVILY_API_KEY")
            print(f"DEBUG: Tavily API key found: {api_key[:8] if api_key else 'None'}...")
            self.client = TavilyClient(api_key=api_key)
            if api_key:
                self.enabled = True
                print(f"DEBUG: Web corrective enabled: {self.enabled}")
            else:
                print("DEBUG: No API key, web corrective disabled")
        except Exception as e:
            print(f"DEBUG: Error initializing Tavily client: {e}")
            self.client = None
            self.enabled = False

    def search(self, question: str, contexts: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[str]]:
        if not self.enabled or not self.client:
            print("DEBUG: Web corrective not enabled or no client")
            return [], []
        
        # Only add context hint if we have meaningful context, but keep it very short
        if contexts and contexts[0].get("text", "").strip():
            # Take only first 100 characters and clean them up
            first_ctx = contexts[0]["text"][:100].strip()
            # Remove special characters and keep only alphanumeric + spaces
            first_ctx = " ".join(word for word in first_ctx.split() if word.isalnum() or word.isspace())[:80]
            if first_ctx:
                q = f"{question}. Focus on market size, competitors, pricing, regulation. Context: {first_ctx}"
            else:
                q = f"{question}. Focus on market size, competitors, pricing, regulation."
        else:
            # Clean query without context hint when no context available
            q = f"{question}. Focus on market size, competitors, pricing, regulation."
        
        print(f"DEBUG: Search query: {q}")
        
        try:
            print("DEBUG: Calling Tavily search...")
            res = self.client.search(q, search_depth="advanced", max_results=5)
            print(f"DEBUG: Tavily response: {res}")
            print(f"DEBUG: Response keys: {list(res.keys()) if res else 'None'}")
        except Exception as e:
            print(f"DEBUG: Tavily search error: {e}")
            return [], []
        
        extra_contexts, source_urls = [], []
        results = res.get("results", [])
        print(f"DEBUG: Found {len(results)} results")
        
        for i, item in enumerate(results):
            print(f"DEBUG: Processing result {i}: {item}")
            url, content = item.get("url",""), item.get("content","")
            if not content:
                print(f"DEBUG: Result {i} has no content, skipping")
                continue
            extra_contexts.append({"text": content[:1800], "metadata": {"source": url, "chunk_index": 0}})
            source_urls.append(url)
            print(f"DEBUG: Added result {i}: {url[:50]}...")
        
        print(f"DEBUG: Final results: {len(extra_contexts)} contexts, {len(source_urls)} sources")
        return extra_contexts[:4], source_urls
