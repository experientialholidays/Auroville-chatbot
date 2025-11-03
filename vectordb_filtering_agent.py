# vectordb_filtering_agent.py

from datetime import datetime
import os
import logging
from typing import Optional, Dict, Any, List

# 💡 Make sure these custom imports are correct for your project structure 💡
from vector_db import VectorDBManager 
from agents import Agent, function_tool, OpenAIChatCompletionsModel
from openai import AsyncOpenAI  # Fixes NameError: AsyncOpenAI

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ----------------- CONFIGURATION -----------------
# Define these variables (Fixes NameError: DB_FOLDER, VECTOR_DB_NAME)
VECTOR_DB_NAME = "vector_db"
DB_FOLDER = "input" 

# ----------------- VECTOR DATABASE SETUP -----------------
db_manager = VectorDBManager(folder=DB_FOLDER, db_name=VECTOR_DB_NAME)
# NOTE: Set force_refresh=True the first time you run this after adding metadata!
vectorstore = db_manager.create_or_load_db(force_refresh=False) 
retriever = db_manager.get_retriever(k=50) 

# ----------------- LLM & AGENT SETUP -----------------
MODEL = "gemini-2.5-flash" 
google_api_key = os.getenv('GOOGLE_API_KEY')

GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
gemini_client = AsyncOpenAI(base_url=GEMINI_BASE_URL, api_key=google_api_key)
gemini_model = OpenAIChatCompletionsModel(model=MODEL, openai_client=gemini_client)

# ----------------- AGENT INSTRUCTIONS -----------------
# Ensure 'datetime' is imported: from datetime import datetime

INSTRUCTIONS = f"""
You will receive:
- user_query: The original user's question
- specificity: Either "Broad" or "Specific", already determined by a previous step.
- raw_results: Raw event data from the vector database (This will be the output of the function call.)

Before calling the function tool `search_auroville_events`, 
first confirm whether a value for `specificity` has been received from the input.

If it was received, state clearly in your reasoning (and optionally your response) 
e.g., "✅ Specificity received from previous step: Broad".
If it was not received, state "⚠️ Specificity not provided — defaulting to 'Broad'".

When calling the tool, always pass the correct `specificity` value.
If missing, default to 'Broad'.

You are an **analytical assistant** with deep knowledge of events and activities happening in Auroville, India.  
Today's date is {datetime.now().strftime("%A, %B %d, %Y, %I:%M %p")}.

**Your final output must be the formatted and filtered list of events. DO NOT include the raw vector db search results in your final output to the user.**

### Style and Behavior Rules
* **Tone and Style:** Maintain a clear, professional, and respectful tone.
* **Deterministic Behavior:** Simulate low-temperature reasoning (0–0.1).
* **Override Defaults:** You are a data extractor, not a conversational partner.
Your goal is to ensure that users can easily discover what's happening in Auroville without being overwhelmed with unnecessary information.

### Final Review Mandate (Self-Correction Step)
Before generating the final response, perform a final self-correction. Review your drafted output against the critical rules for completeness, formatting, and behavior. Revise if necessary.

---

### **Event Data Processing and Formatting Rules**

Events and activities in Auroville occur at three levels:
1. **Date-specific** — Events scheduled for a particular calendar date.
2. **Weekday-based** — Events that happen on recurring weekdays (e.g., every Monday, Wednesday, etc.).
3. **Appointment/daily-based** — Events that require prior booking or appointment or happen daily or from Monday to Friday/Saturday.

#### **Sorting and Grouping**

1. **Group events as per the three levels defined above.**
2. **Sort Chronologically:** The final presentation must follow this order:
    * First, **Date-specific** events.
    * Then, **Weekday-based** events.
    * Finally, **Appointment/daily-based** events.
    * **Internally within each group, sort events by their start time.**

#### **Output Format Selection (Specificity-Based Rule)**

3. **Select Output Format based on Specificity:**
    * **If specificity is "Broad" (Smart Grouping Mode):** Present a concise, clean, numbered list. **Group events with the same contact or type** to avoid overwhelming the user. The **Strict Event Structure is NOT required** for this mode; use a summary format.
    * **If specificity is "Specific" (Detailed Numbered List):** Present all relevant events in a numbered list, following the **Strict Event Structure** defined below.

---

#### **Strict Event Structure (Used ONLY when specificity is "Specific")**

Format each event precisely as follows:

1. **Event Name** - [Short, summarized description]
2. **When**: [Day, Date, and Time]
3. **Where**: [Venue / Location]
4. **Contribution**: [Stated cost, contribution, or entry fee]
5. **Contact**: [Contact Person, Phone, Email, Website/Links]. If a mobile number is provided, generate a **WhatsApp click-to-chat link** with the template message: "Hi, I came across your event '[Event Name]' scheduled on [Event Date]. I would like to request more information. Thank you for your assistance. Best regards,".
6. **Note**: [Special instructions or prerequisites.]
7. **Interactive Details Link**: Generate the command text **[Show details for event #N]** (where N is the event's number in the final list) if the event has a description or poster. **This command text should be formatted as a click-to-chat/click-to-post button/link, so that when the user selects it, the command text itself is placed directly into the user's input/command line.** When the user submits this command, you will fetch and show the full description text. If a poster link is available in the event data, you **MUST** display the poster as an image inline with the description.
"""


# ----------------- RAG TOOL WITH CORRECTED METADATA FILTERING -----------------
@function_tool
def search_auroville_events(
    search_query: str, 
    specificity: str,
    filter_day: Optional[str] = None,      # Metadata filter for day
    filter_date: Optional[str] = None,     # Metadata filter for date
    filter_location: Optional[str] = None  # Metadata filter for location
) -> str:
    """
    Search for information about events and activities. 
    
    If `specificity` is "Broad", the search will include metadata filters (day OR date OR location) to maximize event discovery using OR logic.
    
    Args:
        search_query: The search query about Auroville events (e.g., 'yoga classes').
        specificity: Broad or specfic as per input.
        filter_day: Optional. The specific day of the week to filter by (e.g., 'Monday').
        filter_date: Optional. The specific date to filter by (e.g., 'October 26').
        filter_location: Optional. The location or venue to filter by (e.g., 'Town Hall').
        
    Returns:
        str: Relevant information about Auroville events
    """
    logger.info(f"RAG Tool called with query: {search_query}")
    
    # Dynamically adjust retrieval depth
    k_value = 100 if specificity.lower() == "broad" else 20
    
    # 1. Collect all provided filter values
    chroma_filter: Dict[str, Any] = {}
    simple_filters: Dict[str, str] = {} 

    # --- Enhanced filter handling: automatically include weekday if date is provided ---
    if filter_date:
     simple_filters["date"] = filter_date

    # Try to also derive day of week from date string (if possible)
    if filter_date:
     try:
        parsed_date = datetime.strptime(filter_date, "%B %d, %Y")  # e.g. "October 29, 2025"
        derived_day = parsed_date.strftime("%A")
        simple_filters["day"] = derived_day
        logger.info(f"[FILTER] Derived weekday '{derived_day}' from date '{filter_date}'")
     except ValueError:
        # If date format doesn't include year, try a fallback (e.g. "October 29")
        try:
            parsed_date = datetime.strptime(filter_date + f", {datetime.now().year}", "%B %d, %Y")
            derived_day = parsed_date.strftime("%A")
            simple_filters["day"] = derived_day
            logger.info(f"[FILTER] Derived weekday '{derived_day}' from partial date '{filter_date}'")
        except ValueError:
            logger.warning(f"[FILTER] Could not parse date '{filter_date}' to derive day")
    else:
        logger.info("[FILTER] No date provided — skipping date parsing.")
        
    # Add explicit filters if provided
    if filter_day:
        simple_filters["day"] = filter_day
    if filter_location:
        simple_filters["location"] = filter_location
    # 2. Build the Chroma filter structure for OR logic using $contains
    if len(simple_filters) >= 1:
        # Build the list of individual conditions
        conditions: List[Dict[str, Dict[str, str]]] = []
        for key, value in simple_filters.items():
            conditions.append({key: {"$eq": value}}) 
        
        if len(simple_filters) == 1:
            key = next(iter(simple_filters.keys()))
            chroma_filter[key] = conditions[0][key]
        else:
            chroma_filter["$or"] = conditions
    
    # 3. Prepare search arguments and invoke retriever
    search_kwargs = {"k": k_value}
    
    if chroma_filter:
        search_kwargs["filter"] = chroma_filter 
        logger.info(f"Applying Chroma Filter (OR logic, $contains): {chroma_filter}")

    # Retrieve relevant documents with the vector search and metadata filter
    docs = retriever.invoke(search_query, **search_kwargs)

    # 4. Format Output
    if not docs:
        return "No relevant information found about Auroville events based on your query and filters."
    
    # Format all retrieved documents, displaying the metadata fields for verification
    context = "\n\n".join([
        f"Document {i+1} (Day: {doc.metadata.get('day', 'N/A')} | Date: {doc.metadata.get('date', 'N/A')} | Location: {doc.metadata.get('location', 'N/A')}):\n{doc.page_content}" 
        for i, doc in enumerate(docs)
    ])
    
    logger.info(f"Retrieved {len(docs)} documents for RAG context")
    
    return f"Here is relevant information about Auroville events:\n\n{context}"

# ----------------- AGENT INITIALIZATION -----------------
tools = [search_auroville_events]
    
vectordb_filtering_agent = Agent(
    name="vectordb_query_selector_agent", 
    instructions=INSTRUCTIONS,
    tools=tools,
    model=gemini_model
)
