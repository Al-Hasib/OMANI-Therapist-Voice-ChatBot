from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.utilities import GoogleSerperAPIWrapper
from langchain_community.tools import GoogleSerperRun
from src.config.settings import settings
from typing import Union, Literal

class SearchToolFactory:
    """Factory for creating search tools"""
    
    _tavily_instance = None
    _serper_instance = None
    
    @classmethod
    def get_search_tool(cls, provider: Literal["tavily", "serper"] = "tavily") -> Union[TavilySearchResults, GoogleSerperRun]:
        """Get or create search tool instance (singleton pattern)"""
        if provider == "tavily":
            if cls._tavily_instance is None:
                cls._tavily_instance = TavilySearchResults(
                    api_key=settings.TAVILY_API_KEY
                )
            return cls._tavily_instance
        elif provider == "serper":
            if cls._serper_instance is None:
                search_wrapper = GoogleSerperAPIWrapper(
                    serper_api_key=settings.SERPER_API_KEY
                )
                cls._serper_instance = GoogleSerperRun(api_wrapper=search_wrapper)
            return cls._serper_instance
        else:
            raise ValueError(f"Unsupported provider: {provider}")
    
    @classmethod
    def create_new_search_tool(cls, provider: Literal["tavily", "serper"] = "tavily", **kwargs) -> Union[TavilySearchResults, GoogleSerperRun]:
        """Create a new search tool instance with custom parameters"""
        if provider == "tavily":
            return TavilySearchResults(
                api_key=kwargs.get("api_key", settings.TAVILY_API_KEY),
                max_results=kwargs.get("max_results", settings.SEARCH_RESULTS_COUNT),
                search_depth=kwargs.get("search_depth", settings.TAVILY_SEARCH_DEPTH),
                include_answer=kwargs.get("include_answer", settings.TAVILY_INCLUDE_ANSWER),
                include_raw_content=kwargs.get("include_raw_content", settings.TAVILY_INCLUDE_RAW_CONTENT),
                **{k: v for k, v in kwargs.items() if k not in ["api_key", "max_results", "search_depth", "include_answer", "include_raw_content"]}
            )
        elif provider == "serper":
            search_wrapper = GoogleSerperAPIWrapper(
                serper_api_key=kwargs.get("api_key", settings.SERPER_API_KEY),
                k=kwargs.get("k", settings.SEARCH_RESULTS_COUNT),
                type=kwargs.get("type", settings.SERPER_SEARCH_TYPE),
                country=kwargs.get("country", settings.SERPER_COUNTRY),
                location=kwargs.get("location", settings.SERPER_LOCATION),
                **{k: v for k, v in kwargs.items() if k not in ["api_key", "k", "type", "country", "location"]}
            )
            return GoogleSerperRun(api_wrapper=search_wrapper)
        else:
            raise ValueError(f"Unsupported provider: {provider}")
    
    @classmethod
    def get_tavily_search(cls) -> TavilySearchResults:
        """Convenience method to get Tavily search tool"""
        return cls.get_search_tool("tavily")
    
    @classmethod
    def get_serper_search(cls) -> GoogleSerperRun:
        """Convenience method to get Serper search tool"""
        return cls.get_search_tool("serper")
    
    @classmethod
    def reset_instances(cls):
        """Reset singleton instances (useful for testing)"""
        cls._tavily_instance = None
        cls._serper_instance = None