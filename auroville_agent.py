import os
import logging
import urllib.parse
from datetime import datetime, time, date
from typing import Optional, Dict, Any, List
import re
import pytz
import ast
import shutil   # <--- Keep this
import zipfile  # <--- Keep this
from google.cloud import storage # <--- Keep this

from vectordb_query_selector_agent import vectordb_query_selector_agent
from agents import Agent, function_tool, OpenAIChatCompletionsModel
from vector_db import VectorDBManager, GLOBAL_EVENT_DB
from openai import AsyncOpenAI
from langchain_core.documents import Document
   
logger = logging.getLogger(__name__)
# Create this at the very top of your file (outside any function)
LATEST_DB_TIMESTAMP = None 

# -------------------------------------------------------------------------
# Date/Time Helper Functions
# -------------------------------------------------------------------------

def is_date_specific(date_str, day_str):
    """Classifies an event as date-specific."""
    return bool(date_str and str(date_str).strip().lower() not in ('', 'n/a', 'upcoming', 'none'))

def _parse_date_string(date_str: str, year: int) -> Optional[date]:
    """Robustly parses a date string into a datetime.date object."""
    if not date_str:
        return None
    DATE_FORMATS = ["%B %d, %Y", "%B %d", "%d %B", "%d %b", "%d.%m.%y", "%d.%m.%Y", "%Y-%m-%d", "%d %b %Y", "%d %B %Y"]
    
    for fmt in DATE_FORMATS:
        try:
            p = date_str
            if "%Y" not in fmt and "%y" not in fmt:
                p = f"{date_str.strip()}, {year}"
            return datetime.strptime(p.strip(), fmt.strip()).date()
        except:
            continue
    return None

def parse_time_range(raw: str, title: str = "Unknown", event_date: str = "Unknown") -> tuple[time, time]:
    """
    The main time parser. Extracts (start_time, end_time).
    """
    if not raw:
        return time(0, 0, 0), time(23, 59, 59)

    # Normalize dashes and remove dots from A.M./P.M. for the regex to work
    s = str(raw).replace("—", "-").replace("–", "-").replace(".", "").strip().upper()
    s = re.sub(r'(:\d{2}):\d{2}', r'\1', s)


    if re.search(r'\bANYTIME\b|\bOPEN\b|\bALL DAY\b', s):
        return time(0, 0, 0), time(23, 59, 59)

    pattern = re.compile(r'\b(\d{1,2})(?::(\d{2}))?\s*(AM|PM)?\b')
    matches = list(pattern.finditer(s))

    parsed_times = []
    for i, m in enumerate(matches):
        h = int(m.group(1))
        min_val = int(m.group(2)) if m.group(2) else 0
        meridian = m.group(3)

        if not meridian and i < len(matches) - 1:
            next_meridian = matches[i+1].group(3)
            if next_meridian:
                meridian = next_meridian

        if meridian == "PM" and h != 12:
            h += 12
        if meridian == "AM" and h == 12:
            h = 0
        
        if 0 <= h <= 23 and 0 <= min_val <= 59:
            parsed_times.append(time(h, min_val))

    # Fix: Ensure we don't crash if no times were parsed
    if not parsed_times:
        # Scenario: No start time mentioned
        start_t = time(0, 0, 0)
        end_t = time(23, 59, 59)
    elif len(parsed_times) == 1:
        # Scenario: Start time mentioned, but no end time
        start_t = parsed_times[0]
        # Calculate +1 hour
        new_hour = (start_t.hour + 1) % 24
        end_t = time(new_hour, start_t.minute)
        
        # Special case: if it rolls over to 00:00 (midnight), set to 23:59 to keep it on same day
        if new_hour == 0 and start_t.hour == 23:
            end_t = time(23, 59, 59)
    else:
        # Scenario: Range provided
        start_t = parsed_times[0]
        end_t = parsed_times[1]


    return start_t, end_t

def parse_time_for_sort(raw: str, title: str = "Unknown", event_date: str = "Unknown") -> time:
    """Uses the range parser to get the start time for sorting purposes."""
    start_t, _ = parse_time_range(raw, title=title, event_date=event_date)
    return start_t

# -------------------------------------------------------------------------
# 1. NEW: DOWNLOADER FUNCTION (Must be defined before DB init)
# -------------------------------------------------------------------------

def download_and_unzip_db(bucket_name, local_db_dir):
    global LATEST_DB_TIMESTAMP
    
    if not bucket_name:
        return False

    try:
        storage_client = storage.Client()
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob("latest_vector_db.zip")

        # 1. Light check: Get metadata from Google (very fast)
        blob.reload() 
        remote_time = blob.updated 

                # 2. Compare: If remote is NOT newer, stop here.
        if LATEST_DB_TIMESTAMP and remote_time <= LATEST_DB_TIMESTAMP:
             # This log confirms the code is choosing NOT to download
             logger.info(f"🟢 DB Match: Local version is up-to-date. (Local TS: {LATEST_DB_TIMESTAMP} | Remote TS: {remote_time})")
             return False 

        # 3. New file found! Download and Unzip.
        if LATEST_DB_TIMESTAMP is None:
            logger.info(f"📥 First run: Downloading initial DB from Cloud...")
        else:
            logger.info(f"🔄 Update found: Remote ({remote_time}) is newer than Local ({LATEST_DB_TIMESTAMP}).")
            
        logger.info(f"⚡ Syncing new DB from Cloud...")
        zip_path = "temp_db.zip"
        blob.download_to_filename(zip_path)

        if os.path.exists(local_db_dir):
            shutil.rmtree(local_db_dir)
        os.makedirs(local_db_dir)
        
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(local_db_dir)
            
        if os.path.exists(zip_path):
            os.remove(zip_path)

        # 4. Remember this timestamp for next time
        LATEST_DB_TIMESTAMP = remote_time
        return True # Signals that a REFRESH is needed

    except Exception as e:
        logger.error(f"❌ DB Sync Failed: {e}")
        return False


# -------------------------------------------------------------------------
# 2. Setup & Global Cache
# -------------------------------------------------------------------------

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

VECTOR_DB_NAME = "vector_db"
# CRITICAL CHANGE: Set this to None so it NEVER looks for Excels
DB_FOLDER = None 
MODEL = "gemini-2.5-flash"
GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
google_api_key = os.getenv('GOOGLE_API_KEY')
GCS_BUCKET_NAME = os.getenv("GCS_BUCKET_NAME")

# --- EXECUTE DOWNLOAD NOW (Before Manager Init) ---
if GCS_BUCKET_NAME:
    download_and_unzip_db(GCS_BUCKET_NAME, VECTOR_DB_NAME)

# Now init the manager. 
# It will find the folder we just unzipped.
# It will NOT find an input folder (because DB_FOLDER is None).
db_manager = VectorDBManager(folder=DB_FOLDER, db_name=VECTOR_DB_NAME)

# force_refresh=False is VITAL. We just downloaded a pre-built DB. We don't want to rebuild it.
vectorstore = db_manager.create_or_load_db(force_refresh=False)
retriever = None

def initialize_retriever(vectorstore):
    global retriever
    if vectorstore:
        retriever = vectorstore.as_retriever(search_kwargs={ "k": 50})
    else:
        logger.error("Retriever init failed: Vectorstore is None.")

initialize_retriever(vectorstore)

gemini_client = AsyncOpenAI(base_url=GEMINI_BASE_URL, api_key=google_api_key)
# In your model setup:
gemini_model = OpenAIChatCompletionsModel(
    model=MODEL, 
    openai_client=gemini_client
)


# -------------------------------------------------------------------------
# 2. Formatting Helpers
# -------------------------------------------------------------------------

def format_event_card(doc_metadata: Dict, doc_content: str) -> str:
    title = doc_metadata.get('title', 'Event').strip()
    date_str = doc_metadata.get('date', '').strip()
    raw_day = doc_metadata.get('day', '').strip() 
    day_str = re.sub(r"[\[\]'\",]", " ", raw_day)
    day_str = re.sub(r"\s+", " ", day_str).strip()

    time_str = doc_metadata.get('time', '').strip()
    location = doc_metadata.get('location', '').strip()
    contribution = doc_metadata.get('contribution', '').strip()
    website_link = doc_metadata.get('website_link', '').strip() # <--- GET LINK
    contact_info = doc_metadata.get('contact', '').strip()
    poster_url = doc_metadata.get('poster_url')
    phone_number = doc_metadata.get('phone', '').strip()
    category = doc_metadata.get('category', '').strip()
    description_meta = doc_metadata.get('description', '').strip()
    email_str = doc_metadata.get('email', '').strip()
    audience_str = doc_metadata.get('audience', '').strip()
    

    out = [f"**Event Name:** {title}"]
    if category: out.append(f"**Category:** {category}")

    when = []
    if date_str: when.append(date_str)
    if time_str: when.append(f"@ {time_str}")
    if when: out.append(f"**When:** {' '.join(when)}")

    if location: out.append(f"**Where:** {location}")
    if website_link:
        # Check if it starts with http, if not add it
        link_url = website_link if website_link.startswith("http") else f"https://{website_link}"
        out.append(f"**Website/Link:** [Click here to visit]({link_url})")

    if contribution: out.append(f"**Contribution:** {contribution}")
    if audience_str: out.append(f"**Target Audience/Prerequisites:** {audience_str}")

    contact_parts = []
    if contact_info: contact_parts.append(contact_info)
    if email_str: contact_parts.append(f"Email: {email_str}")

    clean_phone = ''.join(filter(str.isdigit, phone_number))
    wa_link = ""
    if clean_phone:
        msg = f"Hi, I came across your event '{title}' scheduled on {date_str}. May I please have more information about it?"
        wa = urllib.parse.quote(msg)
        wa_link = f"[**Click to Chat on WhatsApp**](https://wa.me/{clean_phone}?text={wa})"
        contact_parts.append(wa_link)

    if contact_parts:
        display_contact = " | ".join([p for p in contact_parts if p != wa_link])
        if wa_link and wa_link in contact_parts:
             display_contact += f"\n{wa_link}"
        out.append(f"**Contact:** {display_contact}")

    out.append("\n**Description:**")
    out.append(description_meta if description_meta else "No detailed description provided.")

    if poster_url:
        out.append(f"\n\n<a href='{poster_url}' target='_blank'>![Event Poster]({poster_url})</a>")

    return "\n".join(out)

def format_summary_numbered(index: int, meta: Dict) -> str:
    title = meta.get('title', '').strip()
    date_str = meta.get('date', '').strip()
    raw_day = meta.get('day', '').strip()
    day = re.sub(r"[\[\]'\",]", " ", raw_day)
    day = re.sub(r"\s+", " ", day).strip()
    time_str = meta.get('time', '').strip()
    loc = meta.get('location', '').strip()
    contrib = meta.get('contribution', '').strip()
    phone = meta.get('phone', '').strip()
    audience = meta.get('audience', '').strip()

    parts = []
    if date_str: parts.append(date_str) 
    elif day: parts.append(day) 
    if time_str: parts.append(time_str)
    if loc: parts.append(f"@{loc}")
    if contrib: parts.append(f"| Contrib: {contrib}")
    if phone: parts.append(f"| Ph:{''.join(filter(str.isdigit, phone))}")
    if audience: parts.append(f"| Audience: {audience}")

    event_id = meta.get("uuid", "").strip()
    return (f"{index}. **{title}** — {' '.join(parts)}\n"
            f"   👉 <a href='#DETAILS::{event_id}'>View details</a>")

# -------------------------------------------------------------------------
# 3. Tools 
# -------------------------------------------------------------------------

def get_daily_events_core(start_number: int) -> str:
    global vectorstore
    try:
        raw = vectorstore.get(where={"category": "Daily Events"}, include=["documents", "metadatas"])
    except Exception as e:
        return f"Error fetching daily events: {e}"

    if not raw or not raw.get("ids"):
        return "No Daily Events found."

    docs = [Document(page_content=text, metadata=meta) for text, meta in zip(raw["documents"], raw["metadatas"])]
    for d in docs:
        d.metadata["_sort_time"] = parse_time_for_sort(
            d.metadata.get("time", ""), 
            title=d.metadata.get("title", "Daily Event"),
            event_date="Daily"
        )

    docs.sort(key=lambda d: d.metadata["_sort_time"])
    out_lines = ["\n## 🌞 Daily Events"]
    idx = start_number
    for d in docs:
        idx += 1
        out_lines.append(format_summary_numbered(idx, d.metadata))
        out_lines.append("")
    return "\n".join(out_lines)

@function_tool
def get_daily_events(start_number: int) -> str:
    return get_daily_events_core(start_number)

@function_tool
def search_auroville_events(search_query: str, specificity: str, filter_day: Optional[str] = None, filter_date: Optional[str] = None, filter_location: Optional[str] = None) -> str:
    global retriever, vectorstore

    # 1. This checks GCS. Only returns True if a NEW zip was downloaded.
    did_update = download_and_unzip_db(GCS_BUCKET_NAME, VECTOR_DB_NAME)

    # 2. Only reload the "In-Memory" database if the files actually changed.
    if did_update:
        logger.info("🔄 Files changed on GCS. Reloading Python Vectorstore...")
        vectorstore = db_manager.create_or_load_db(force_refresh=False)
        initialize_retriever(vectorstore) # This resets the 'retriever' global

    # 3. Proceed with search as normal
    if retriever is None: 
        return "Database is currently updating. Please try again in 5 seconds."

    k_value = 325 if specificity.lower() == "broad" else 30
    or_conditions = []
    query_date_list = [] 
    TIMEZONE_AV = pytz.timezone('Asia/Kolkata')
    now_dt = datetime.now(TIMEZONE_AV)
    today = now_dt.date()
    now_time = now_dt.time() 
   
    SPLIT_PATTERN = re.compile(r'\s*,\s*|\s+(?i)to\s+')

    if filter_day:
        days = [d.strip().title() for d in SPLIT_PATTERN.split(filter_day) if d.strip()]
        for d in days: or_conditions.append({"day": {"$eq": d}}) 

    
    if filter_date:
        dates = [d.strip() for d in SPLIT_PATTERN.split(filter_date) if d.strip()]
        for d in dates:
            # 1. POPULATE THE LIST (The most important part)
            try:
                # This turns the text into a date object Python can do math with
                parsed_date = datetime.strptime(d, "%Y-%m-%d").date()
                query_date_list.append(parsed_date)
            except Exception as e:
                logger.error(f"Error parsing filter date {d}: {e}")

            # 2. ADD CHROMA FILTERS (The "Broad Search" part)
            # Exact ISO match
            or_conditions.append({"start_date_meta": {"$eq": d}})
            
            try:
                date_obj = datetime.strptime(d, "%Y-%m-%d").date()
                # Weekday match (e.g., "Sunday")
                or_conditions.append({"day": {"$eq": date_obj.strftime("%A")}})
                # Human readable text match
                readable_date = date_obj.strftime("%-d %B %Y")
                or_conditions.append({"date": {"$eq": readable_date}})
            except:
                pass


    chroma_filter = or_conditions[0] if len(or_conditions) == 1 else {"$or": or_conditions} if or_conditions else {}
    docs = retriever.invoke(search_query, **{"k": k_value, "filter": chroma_filter} if chroma_filter else {"k": k_value})

    raw_count = len(docs) if docs else 0
    logger.info(f"\n🔍 [GATE 1: CHROMA] Found {raw_count} raw events.")
    if raw_count > 0:
        sample_meta = docs[0].metadata
        ts_val = sample_meta.get('start_timestamp')
        logger.info(f"📡 [DATA TYPE CHECK] start_timestamp type: {type(ts_val)} | value: {ts_val}")
        logger.info(f"📡 [FILTER SENT]: {chroma_filter}")
    
    if not docs: return "I couldn't find any upcoming events matching those criteria."

    filtered = []
    seen = set()
    seen_fingerprints = set() # <--- NEW: To track Name + Date + Time
   
    for doc in docs:
        event_uuid = doc.metadata.get("uuid")
        if event_uuid in seen: continue
           
        source_file = doc.metadata.get('source', 'Unknown_Source')
        title = doc.metadata.get('title', 'Unknown')
        time_str = str(doc.metadata.get('time', '')).strip()
        start_str = str(doc.metadata.get('start_date_meta', '')).strip()
        end_meta_str = str(doc.metadata.get('end_date_meta', '')).strip()
        day_val = str(doc.metadata.get('day', '')).strip()
        
        doc_start_date = _parse_date_string(start_str, today.year)
        doc_end_date = _parse_date_string(end_meta_str, today.year)
        start_t, end_t = parse_time_range(time_str, title=title)

        logger.info(f"🛠️ [RAG PROCESSING] Event: {title} | Dates: {doc_start_date} to {doc_end_date} | Times: {start_t} to {end_t} from {source_file}")

        event_date_key = start_str if start_str else day_val
        fingerprint = f"{title}|{event_date_key}|{time_str}"

        if fingerprint in seen_fingerprints:
            logger.info(f" ⏩ Skipping duplicate event: {title} at {time_str}")
            continue

        # --- 1. DATE FILTERING (REFACTORED) ---
        if query_date_list:
            is_match = False
            q_min = min(query_date_list)
            q_max = max(query_date_list)
            d_start = doc_start_date
            d_end = doc_end_date if doc_end_date else doc_start_date

            # Check for range overlap
            if d_start:
                if d_start <= q_max and d_end >= q_min:
                    is_match = True
            
            # Check for recurring day match if no specific date on event
            if not is_match and d_start is None and filter_day and day_val:
                user_days = [d.strip().title() for d in SPLIT_PATTERN.split(filter_day) if d.strip()]
                if day_val in user_days:
                    is_match = True

            # Logging and skipping
            if is_match:
                logger.info(f" ✅ MATCH: '{title}' ({d_start} to {d_end}) overlaps query ({q_min} to {q_max})")
            else:
                logger.info(f" ❌ REJECT: '{title}' ({d_start} to {d_end}) is outside query ({q_min} to {q_max})")
                continue # Safely moves to the next doc in the loop

        # --- 2. EXPIRATION FILTERING ---
        eff_end = doc_end_date if doc_end_date else doc_start_date
        if eff_end and eff_end < today:
            logger.info(f" ❌ REJECT: '{title}' already ended on {eff_end}")
            continue

        # --- 3. TODAY'S TIME FILTERING ---
        is_happening_on_target_day = False
        
        # Determine if the event is valid for "Today" (the current wall-clock date)
        if doc_start_date and doc_end_date and (doc_start_date <= today <= doc_end_date):
            is_happening_on_target_day = True
        elif not doc_start_date and day_val:
            if day_val.lower() == today.strftime("%A").lower():
                is_happening_on_target_day = True

        if is_happening_on_target_day:
            # FIX: Only reject based on time if the user is SEARCHING for today.
            # If query_date_list is empty, we assume a general "What's on" (Today) search.
            # If query_date_list has dates, check if 'today' is one of them.
            searching_for_today = False
            if not query_date_list:
                searching_for_today = True
            else:
                searching_for_today = any(d == today for d in query_date_list)

            # Only apply the 6 PM / 8 PM 'already ended' logic if viewing today's results
            if searching_for_today and now_time > end_t:
                logger.info(f" ❌ REJECT: Finished for today: {title} (Ended at {end_t})")
                continue


        # --- 4. FINAL ADDITION ---
        full_doc = GLOBAL_EVENT_DB.get(event_uuid)
        if full_doc:
            filtered.append(full_doc)
            seen.add(event_uuid)
            seen_fingerprints.add(fingerprint)

    
    logger.info(f"🏁 [GATE 2: FINAL] Python filtered {raw_count} down to {len(filtered)} events.\n")
    for doc in filtered:
        title_meta = doc.metadata.get("title", "Unknown")
        date_meta = doc.metadata.get("date", "Unknown")
        
        # 1. Store the 24h TIME object in a hidden key
        doc.metadata["_sort_time"] = parse_time_for_sort(
            doc.metadata.get("time", ""), 
            title=title_meta, 
            event_date=date_meta
        )

        # 2. Store a sortable DATE object (YYYY-MM-DD)
        # We use start_date_meta because it's usually already '2025-12-27'
        raw_date = str(doc.metadata.get("start_date_meta", "9999-12-31"))
        try:
            doc.metadata["_sort_date"] = datetime.strptime(raw_date, "%Y-%m-%d").date()
        except:
            doc.metadata["_sort_date"] = date(9999, 12, 31) # Fallback for missing dates


        raw = (doc.metadata.get('category') or "").lower()
        if "date" in raw: doc.metadata["category"] = "Date-specific Events"
        elif "week" in raw: doc.metadata["category"] = "Weekly Events"
        elif any(x in raw for x in ["daily", "appoint", "everyday"]): doc.metadata["category"] = "Daily Events"
        else:
            if is_date_specific(doc.metadata.get('date', ''), doc.metadata.get('day', '')): doc.metadata["category"] = "Date-specific Events"
            elif doc.metadata.get('day'): doc.metadata["category"] = "Weekly Events"
            else: doc.metadata["category"] = "Daily Events"

    categories = ["Date-specific Events", "Weekly Events", "Daily Events"]
    buckets = {c: [] for c in categories}
    for d in filtered:
        c = d.metadata.get("category")
        if c in buckets: buckets[c].append(d)

    for c in buckets:
    # This sorts by Date first, then by Time
        buckets[c].sort(key=lambda d: (d.metadata["_sort_date"], d.metadata["_sort_time"]))

    out_lines = []
    idx = 0
    broad = (specificity.lower() == "broad")
    for c in ["Date-specific Events", "Weekly Events"]:
        if buckets[c]:
            out_lines.append(f"\n **{c}**")
            for d in buckets[c]:
                idx += 1
                out_lines.append(format_summary_numbered(idx, d.metadata))
                out_lines.append("")

    if not broad and buckets["Daily Events"]:
        out_lines.append("\n## 🌞 Daily Events")
        for d in buckets["Daily Events"]:
            idx += 1
            out_lines.append(format_summary_numbered(idx, d.metadata))
            out_lines.append("")

    if broad:
        out_lines.append("\nThere are Daily Events also happening every day.\n👉 <a href='#SHOWDAILY::YES'>Yes</a> 👉 <a href='#SHOWDAILY::NO'>No</a>")

    return "\n".join(out_lines)

def get_event_details_core(identifier: str) -> str:
    if not identifier: return "No event identifier provided."
    event_uuid = str(identifier).strip() 
    doc = GLOBAL_EVENT_DB.get(event_uuid) 
    if not doc: return f"I could not find an event for ID {event_uuid}."
    return format_event_card(doc.metadata, doc.page_content)

        
@function_tool
def get_event_details(identifier: str) -> str:
    """Returns the full details for a specific event by UUID."""
    return get_event_details_core(identifier)

INSTRUCTIONS = f"""
You are an AI assistant that answers user questions about Auroville events.
​I. General Query Handling & Tool Execution
​Your primary function is to handle general event queries (e.g., "events tomorrow", "yoga events", "workshops this weekend", "events on 24 November", "sound healing", "children events", "what is happening today", "things to do in Auroville").
​A. Two-Step Search Protocol (MANDATORY)
​For every general search query, you MUST follow this two-step process:
​Refine Query: Do NOT call search_auroville_events directly. You MUST FIRST call vectordb_query_selector_agent to refine the user's search query.
​Execute Search: You MUST THEN call search_auroville_events using the refined query returned from the previous step.
​B. Tool Output Processing & Filtering
​When the search_auroville_events tool returns results:
​Filtering (Strict Rule 1): If an event returned by the tool clearly does not match the user's original query (e.g., a "meditation" event is returned for a "sports" search), you MUST exclude it from the final response.
​Duplicate Removal (Strict Rule 2): You MUST remove all duplicate events from the final list.
​Source Integrity (Strict Rule 3): You MUST treat the tool output as the source and keep its formatting intact.
​No Hallucination (Strict Rule 4): You MUST NOT hallucinate or invent missing event information. If metadata (like time or location) is missing for an event, you MUST omit that specific field; do not replace it with placeholders.
​II. Formatting and Output Rules
​Format Preservation (Strict Rule 5): You MUST keep the output format exactly intact as you received it from the tool.
​Category Preservation (Strict Rule 6): When the tool output lists categories, you MUST keep them exactly as:
​Date-specific Events
​Weekly Events
​Daily Events
​Link Integrity (Strict Rule 7): You MUST NOT modify or break clickable HTML links (e.g., View Details, Yes/No, or any other <a> tag content). These links must remain untouched.
​III. Prohibited Actions (App Intercepts)
​You MUST NOT handle or respond to the following types of input, as they are managed directly by the application code. While you may see them as plain text, you will never be invoked for them:
​details(NUM)
​NUM (e.g., 1, 2, 3)
​show daily events
​IV. Permitted Minor Corrections
​You may only perform the following minor corrections outside of the content blocks returned by the tools (i.e., in your introductory or transitional text):
​Correct small grammar issues.
​Remove duplicate text for better understanding.

"""

tools = [
    vectordb_query_selector_agent.as_tool("vectordb_query_selector_agent", "Refines query."),
    search_auroville_events,
    get_daily_events,
    get_event_details
]

auroville_agent = Agent(
    name="Auroville Events Assistant",
    instructions=INSTRUCTIONS,
    model=gemini_model,
    tools=tools
       )
