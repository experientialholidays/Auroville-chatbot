from datetime import datetime
from agents import Agent, function_tool,OpenAIChatCompletionsModel
from openai import AsyncOpenAI
from vector_db import VectorDBManager
import os
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Vector Database
VECTOR_DB_NAME = "vector_db"
DB_FOLDER = "input"
db_manager = VectorDBManager(folder=DB_FOLDER, db_name=VECTOR_DB_NAME)
vectorstore = db_manager.create_or_load_db(force_refresh=False)
retriever = db_manager.get_retriever(k=50)


# Configuration
MODEL = "gpt-5"
google_api_key = os.getenv('GOOGLE_API_KEY')

GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
gemini_client = AsyncOpenAI(base_url=GEMINI_BASE_URL, api_key=google_api_key)
gemini_model = OpenAIChatCompletionsModel(model="gemini-2.5-flash", openai_client=gemini_client)


INSTRUCTIONS = f""" You will receive:
- user_query: The original user's question
- raw_results: Raw event data from the vector database

You are an analytical assistant with deep knowledge of events and activities happening in Auroville, India.  
Today's date is {datetime.now().strftime("%A, %B %d, %Y, %I:%M %p")}.

Events and activities in Auroville occur at three levels:

1) **Date-specific** — Events scheduled for a particular calendar date.  
2) **Weekday-based** — Events that happen on recurring weekdays (e.g., every Monday, Wednesday, etc.).  
3) **Appointment-based** — Events that require prior booking or appointment.

Your task:
- Carefully interpret the user's query and identify which type(s) of events are relevant.  
- **Always include appointment-based events** when users ask about any date, day, or type of activity, since they may not be aware of these.  
- raw_results might have weekly events comprising only of day of the weeek. You need to consider this if day of the week in user_query is matching with it. 
- Focus on **accuracy and contextual relevance** — only include events that truly match the user's intent.  
- Keep responses **concise and structured**. Do **not** provide detailed descriptions for every event unless the user explicitly requests more details.  
- Maintain a friendly, factual, engaging and clear tone throughout.  

Your goal is to ensure that users can easily discover what's happening in Auroville without being overwhelmed with unnecessary information.
"""

@function_tool
def search_auroville_events(search_query: str,specificity: str = "Broad") -> str:
    """
    Search for information about Auroville events and activities. 
    Use this tool whenever the user asks about events, activities, schedules, or anything related to Auroville.
    
    Args:
        search_query: The search query about Auroville events (e.g., 'dance events in October', 'music workshops', 'yoga classes')
        specificity: Determine query specificity:
                    - Broad (general date/day queries)
                    - Specific (particular event/activity queries)
        
    Returns:
        str: Relevant information about Auroville events
    """
    logger.info(f"RAG Tool called with query: {search_query}")
    
    # Dynamically adjust retrieval depth
    k_value = 50 if specificity.lower() == "broad" else 10
    # Retrieve relevant documents (uses k=50 from retriever config)
    docs = retriever.get_relevant_documents(search_query,k=k_value)

    
    if not docs:
        return "No relevant information found about Auroville events."
    
    # Format all retrieved documents
    context = "\n\n".join([
        f"Document {i+1}:\n{doc.page_content}" 
        for i, doc in enumerate(docs)
    ])
    
    logger.info(f"Retrieved {len(docs)} documents for RAG context")
    
    return f"Here is relevant information about Auroville events:\n\n{context}"

tools = [search_auroville_events]
    
vectordb_filtering_agent = Agent(name="vectordb_query_selector_agent", instructions=INSTRUCTIONS,tools=tools,model=gemini_model)