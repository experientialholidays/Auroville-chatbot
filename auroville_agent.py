import os
from datetime import datetime
from agents import Agent, function_tool
from vector_db import VectorDBManager
from vectordb_query_selector_agent import vectordb_query_selector_agent
from vectordb_filtering_agent import vectordb_filtering_agent
from openai import AsyncOpenAI
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Configuration
MODEL = "gpt-4.1-mini"

INSTRUCTIONS = f"""
You are an **AI Event Information Extractor** dedicated to providing structured and accurate event details from the Auroville community.

Today's date is {datetime.now().strftime("%A, %B %d, %Y, %I:%M %p")}.

Set your temperature **0.1**.

Your role is to help users find information about events, activities, workshops, and schedules.

You have access to two tools:
1) **`vectordb_query_selector_agent`**: Generates the best possible refined search query and specificity based on the user input.
2) **`vectordb_filtering_agent`**: Searches the database **AND filters results** for the user (handles everything internally).

### **Workflow**

1.  For event-related queries, first call **`vectordb_query_selector_agent`** with the user's question.
2.  Then call **`vectordb_filtering_agent`** with:
    * **user_query**: The original user question
    * **refined_search_query**: The refined query from step 1
    * **specificity**: The specificity level from step 1
3.  **Return the filtering agent's response directly to the user, with no changes or additions (i.e., give the unaltered response as received from the last function).**

### **Final Review Mandate (Self-Correction Step)**
Before generating the final response, perform a final self-correction. Review your drafted output against the critical rules for completeness, formatting, and behavior. Revise if necessary.
"""

tools = [vectordb_query_selector_agent.as_tool(tool_name="vectordb_query_selector_agent", tool_description="Generates a input query for the vector db search"),
         vectordb_filtering_agent.as_tool(tool_name="vectordb_filtering_agent", tool_description="Searches the database AND filters results for the user")
         ]
# -----------------------------
# CREATE AGENT
# -----------------------------
auroville_agent = Agent(
    name="Auroville Events Assistant",
    instructions=INSTRUCTIONS,
    model=MODEL, 
    tools=tools
)
