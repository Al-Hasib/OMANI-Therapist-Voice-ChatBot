# from langchain_core.prompts import ChatPromptTemplate
# import json

# class Prompts:
#     """Centralized prompt templates"""
    
#     QUERY_UPGRADER = ChatPromptTemplate.from_messages([
#         ("system", """You are a query enhancement specialist. Your task is to improve user queries for better information retrieval.
        
#         Enhancement guidelines:
#         1. Add relevant keywords and synonyms
#         2. Clarify ambiguous terms
#         3. Expand abbreviations and acronyms
#         4. Add context when missing
#         5. Maintain original intent
#         6. Keep enhanced query concise (under 200 characters)
        
#         Return only the enhanced query, nothing else."""),
#         ("human", "Original query: {query}")
#     ])
    
#     QUERY_ROUTER = ChatPromptTemplate.from_messages([
#         ("system", """You are a query router. Analyze the query and decide which path to take:

#         PATHS:
#         1. "RAG" - For queries about specific knowledge base content, documents, or domain expertise
#         2. "WEB" - For current events, real-time information, recent news, or trending topics
#         3. "DIRECT" - For general conversation, creative tasks, opinions, or reasoning without specific facts

#         DECISION CRITERIA:
#         - RAG: Domain-specific questions, technical documentation, specific facts from knowledge base
#         - WEB: Questions with temporal keywords (latest, current, recent, today), current events, real-time data
#         - DIRECT: General chat, creative writing, opinions, mathematical reasoning, casual conversation

#         Respond with only one word: RAG, WEB, or DIRECT"""),
#         ("human", "Query: {query}")
#     ])
    
#     RAG_RESPONSE = ChatPromptTemplate.from_messages([
#         ("system", """You are a helpful assistant. Answer the user's question based on the provided context from the knowledge base.
        
#         Context: {context}
        
#         If the context doesn't contain relevant information, say so clearly."""),
#         ("human", "{query}")
#     ])
    
#     WEB_RESPONSE = ChatPromptTemplate.from_messages([
#         ("system", """You are a helpful assistant. Answer the user's question based on the provided web search results.
        
#         Search Results: {search_results}
        
#         Provide a comprehensive answer based on the search results. If the results don't contain relevant information, say so clearly."""),
#         ("human", "{query}")
#     ])
    
#     DIRECT_RESPONSE = ChatPromptTemplate.from_messages([
#         ("system", """You are a helpful AI assistant. Answer the user's question directly using your knowledge and reasoning capabilities.
        
#         Be conversational, accurate, and helpful. If you're unsure about something, acknowledge the uncertainty."""),
#         ("human", "{query}")
#     ])

# def load_data_relative():
#     """Load data.json using relative path"""
#     try:
#         with open("knowledge_base_metadata.json", 'r', encoding='utf-8') as f:
#             data = json.load(f)
#         description = ""
#         for key in data:
#             description +=f"{key['description']}\n"
#         return description
#     except FileNotFoundError:
#         print("data.json not found in current directory")
#         return None
#     except json.JSONDecodeError as e:
#         print(f"Error decoding JSON: {e}")
#         return None

# if __name__=="__main__":
#     print(load_data_relative())

from langchain_core.prompts import ChatPromptTemplate
import json

class Prompts:
    """Centralized prompt templates"""

    @staticmethod
    def load_kb_description():
        """Dynamically load knowledge base descriptions"""
        try:
            with open("knowledge_base_metadata.json", 'r', encoding='utf-8') as f:
                data = json.load(f)
            description = ""
            for item in data:
                description += f"- {item.get('description', '').strip()}\n"
            return description.strip()
        except FileNotFoundError:
            return "No knowledge base found."
        except json.JSONDecodeError as e:
            return f"Error decoding knowledge base: {e}"

    @classmethod
    def query_router(cls):
        """Return QUERY_ROUTER with dynamic KB info"""
        kb_description = cls.load_kb_description()
        return ChatPromptTemplate.from_messages([
            ("system", f"""You are a query router. Analyze the query and decide which path to take:

PATHS:
1. "RAG" - For queries about specific knowledge base content, documents, or domain expertise
2. "WEB" - For current events, real-time information, recent news, or trending topics
3. "DIRECT" - For general conversation, creative tasks, opinions, or reasoning without specific facts

DECISION CRITERIA:
- RAG: Domain-specific questions, technical documentation, specific facts from knowledge base
- WEB: Questions with temporal keywords (latest, current, recent, today), current events, real-time data
- DIRECT: General chat, creative writing, opinions, mathematical reasoning, casual conversation

Knowledge Base contains:
{kb_description}

Respond with only one word: RAG, WEB, or DIRECT"""),
            ("human", "Query: {{query}}")
        ])

    # Other prompts stay unchanged
    QUERY_UPGRADER = ChatPromptTemplate.from_messages([
        ("system", """You are a query enhancement specialist..."""),  # shortened for brevity
        ("human", "Original query: {query}")
    ])

    RAG_RESPONSE = ChatPromptTemplate.from_messages([
        ("system", """You are a helpful assistant. Answer the user's question based on the provided context from the knowledge base.
        
        Context: {context}
        
        If the context doesn't contain relevant information, say so clearly."""),
        ("human", "{query}")
    ])

    WEB_RESPONSE = ChatPromptTemplate.from_messages([
        ("system", """You are a helpful assistant. Answer the user's question based on the provided web search results.
        
        Search Results: {search_results}
        
        Provide a comprehensive answer..."""),
        ("human", "{query}")
    ])

    DIRECT_RESPONSE = ChatPromptTemplate.from_messages([
        ("system", """You are a helpful AI assistant. Answer the user's question directly using your knowledge and reasoning capabilities."""),
        ("human", "{query}")
    ])

if __name__ == "__main__":
    prompt = Prompts.query_router()
    print(prompt.format(query="What is the architecture of the Omani AI system?"))
