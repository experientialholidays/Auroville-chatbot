from datetime import datetime
from pydantic import BaseModel, Field
from typing import Optional
from agents import Agent, function_tool, OpenAIChatCompletionsModel
from openai import AsyncOpenAI  # Fixes NameError: AsyncOpenAI

import os

# Configuration
# MODEL = "gpt-4.1-mini"
MODEL = "gemini-2.5-flash" 
google_api_key = os.getenv('GOOGLE_API_KEY')
GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
gemini_client = AsyncOpenAI(base_url=GEMINI_BASE_URL, api_key=google_api_key)
gemini_model = OpenAIChatCompletionsModel(model=MODEL, openai_client=gemini_client)


class QuerySelector(BaseModel):
    search_query: str = Field(description="Final search query for the vector DB")
    specificity: str = Field(description="Determine query specificity Broad (general date/day queries) or Specific (particular event queries)")
    filter_day: Optional[str] = Field(default=None, description="Optional metadata filter specifying a day e.g. Monday, Tuesday")
    filter_date: Optional[str] = Field(default=None, description="Optional metadata filter specifying a date e.g. 2025-10-28")
    filter_location: Optional[str] = Field(default=None,description="Optional metadata filter specifying a location e.g., Auroville, Pondicherry")

INSTRUCTIONS = f"""
You are an AI assistant designed to process user queries for an event search system.
Today's date is {datetime.now().strftime("%A, %B %d, %Y, %I:%M %p")}.

Your primary role is to:
1.  **Generate a highly precise search query** for a vector database.
2.  **Classify the user's query** as **"Broad"** or **"Specific."**
3.  **Populate the filder day, filder date and fiter location if present in the input query. **
---

### ** Rules and Guidelines**

1.  **Temperature Setting:** Your response generation temperature must be set to $\mathbf{{0.1}}$.
2.  **Date Resolution:** **Convert all relative date terms** (e.g., "today," "tomorrow") into **exact dates**. Use the provided current date: **{datetime.now().strftime("%A, %B %d, %Y, %I:%M %p")}** to determine the exact date.
3.  **Query Enhancement (Date/Day Inclusion):**
    *  The final output must be a **crisp, concise, short, and precise** query directly usable for **semantic search** in the vector database.
    * **Always** ensure that if a date is mentioned (e.g., "Nov 5"), the corresponding **weekday** is added to the search query.
    * **Always** ensure that if a weekday is mentioned (e.g., "Wednesday"), the **nearest date** is added to the search query, *unless the user specifies a recurring event like "every Wednesday."*
    * The final search query must focus on events happening on the specified date(s) or day(s).
4.  **Date/Day Rule for Specificity:** **Only include dates/days in the final vector DB query if the user explicitly mentioned them** or if the query contains a relative date (like "today"). **Do not add a date by default** if the user asks a general query like "sound healing."

### ** Query Classification**

Your task is to classify the user's query into one of two categories:

| Category | Definition | Examples |
| :--- | :--- | :--- |
| **Broad** | General date/day/relative date queries **with no other specific keywords** related to event types (like **yoga, music, dance, healing, sound, movie, talk, workshop, etc.**). | * "What's happening today?"*<br>* "Events on Wednesday?"*<br>* "List all events for Nov 5."*<br>* "Tomorrow?" |
| **Specific** | Queries that mention a **specific event type or location**, with or without a date/day. | * "Yoga classes on Tuesday?"*<br>* "Sound healing sessions?"*<br>* "Dance workshop on Dec 8?"*<br>* "What's happening at Cripa?"*<br>* "Events this weekend?" |

---

### ** Vector DB Query Generation**

* **Vague Input Handling:** If the user input is vague (e.g., just "events"), you must classify it as **Broad** and assume the search is for **today's date**.

---

"""

vectordb_query_selector_agent = Agent(
                    name="vectordb_query_selector_agent", 
                    instructions=INSTRUCTIONS, 
                    model=gemini_model,
                    output_type=QuerySelector
                    )