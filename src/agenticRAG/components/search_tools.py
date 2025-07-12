from langchain_community.tools import GoogleSearchRun
from langchain_community.utilities import GoogleSearchAPIWrapper
from config.settings import settings

class SearchToolFactory:
    """Factory for creating search tools"""
    
    _instance = None
    
    @classmethod
    def get_search_tool(cls) -> GoogleSearchRun:
        """Get or create search tool instance (singleton pattern)"""
        if cls._instance is None:
            search_wrapper = GoogleSearchAPIWrapper(
                google_api_key=settings.GOOGLE_API_KEY,
                google_cse_id=settings.GOOGLE_CSE_ID,
                k=settings.SEARCH_RESULTS_COUNT
            )
            cls._instance = GoogleSearchRun(api_wrapper=search_wrapper)
        return cls._instance