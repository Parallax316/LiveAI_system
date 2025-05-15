# main.py
import streamlit as st
import llm_interaction # Your module for LLM calls (ensure it's a version that handles list[dict] for scraped_articles_data)
import google_search_scraper as search_scraper_module # Using Google Search scraper
import content_elaboration # The module for content elaboration
import requests # For IP-based location
import datetime


st.set_page_config(page_title="Live Search Agent with CoT (Google Search)", layout="wide")

st.title("🔎 Live Search Agent with Chain of Thoughts (Google Search)")
st.caption("Enter a query to get a web-informed answer with visible reasoning steps.")

# Fetch user city via IP geo-lookup
@st.cache_data(ttl=3600) # Updated decorator for newer Streamlit versions
def get_user_city():
    try:
        # Increased timeout slightly for robustness
        resp = requests.get('https://ipinfo.io/json', timeout=10)
        resp.raise_for_status() # Raise an exception for HTTP errors
        data = resp.json()
        city = data.get('city', '')
        region = data.get('region', '')
        country = data.get('country', '')
        if city and region and country:
            return f"{city}, {region}, {country}"
        elif city and country:
            return f"{city}, {country}"
        elif city:
            return city
        return "Location not detected"
    except Exception as e:
        st.sidebar.warning(f"Could not detect location: {e}")
        return "Location not detected"

user_location_info = get_user_city()
if user_location_info and "not detected" not in user_location_info:
    st.sidebar.info(f"Detected location: {user_location_info}")
else:
    st.sidebar.warning("User location not automatically detected. Search results may be less localized.")


# --- Helper function to display CoT ---
def display_cot(cot_header, cot_text):
    if cot_text:
        cot_text_str = str(cot_text) if cot_text is not None else "No chain of thought text available."
        # Using st.expander to make it collapsible by default
        with st.expander(cot_header, expanded=False):
            st.markdown(f"```text\n{cot_text_str}\n```")
    else:
        st.warning(f"No {cot_header.lower()} available.")

# --- Main Application Logic ---
user_query = st.text_input("Enter your search query:", placeholder="e.g., What are the latest advancements in AI?")

if 'history' not in st.session_state:
    st.session_state.history = []
if 'current_outline' not in st.session_state: 
    st.session_state.current_outline = None
if 'original_query_for_elaboration' not in st.session_state:
    st.session_state.original_query_for_elaboration = None


if st.button("🚀 Get Live Answer", type="primary"):
    st.session_state.current_outline = None 
    st.session_state.original_query_for_elaboration = None

    if not user_query:
        st.error("Please enter a search query.")
    else:
        st.markdown("---") 
        
        current_process_log = {"query": user_query, "search_provider": "Google Search", "steps": []}
        st.session_state.original_query_for_elaboration = user_query 

        # --- Phase 1: LLM plans the search ---
        st.subheader("Phase 1: Planning the Search")
        search_query_for_scraper = None
        phase1_cot = None
        with st.spinner("Asking LLM to analyze query and suggest a search term..."):
            try:
                search_query_for_scraper, phase1_cot = llm_interaction.get_search_query_and_cot(user_query)
                current_process_log["steps"].append({
                    "name": "LLM Search Plan", 
                    "query_suggestion": search_query_for_scraper, 
                    "cot_snippet": (phase1_cot[:150]+"..." if phase1_cot else "N/A") 
                })
            except Exception as e:
                st.error(f"Error in Phase 1 (LLM planning): {e}")
                phase1_cot = f"An error occurred: {e}"
                current_process_log["steps"].append({"name": "LLM Search Plan Error", "error": str(e)})

        display_cot("📝 LLM's Search Plan (Chain of Thought)", phase1_cot)

        if not search_query_for_scraper:
            st.error("LLM could not determine a search query. Please try rephrasing your input.")
        else:
            st.info(f"🔍 LLM suggested base search query for Google Search: **`{search_query_for_scraper}`**")
            st.markdown("---")

            # --- Phase 2: Web Scraping using Google Search ---
            st.subheader("Phase 2: Retrieving Web Content via Google Search")
            # This will be a list of dictionaries from get_content_from_google_search
            scraped_articles_data = [] 

            # Determine lookback_hours based on query - simple heuristic for now
            lookback_hours_for_query = None
            if "last 24 hours" in user_query.lower() or "today" in user_query.lower():
                lookback_hours_for_query = 24
            elif "last week" in user_query.lower():
                lookback_hours_for_query = 24 * 7
            
            location_for_search = user_location_info if "not detected" not in user_location_info else None


            with st.spinner(f"Searching Google for '{search_query_for_scraper}' (Location: {location_for_search or 'N/A'}, Lookback: {lookback_hours_for_query or 'N/A'}h) and scraping articles..."):
                try:
                    scraped_articles_data = search_scraper_module.get_content_from_google_search(
                        search_query_for_scraper,
                        location=location_for_search,
                        lookback_hours=lookback_hours_for_query
                    )
                    current_process_log["steps"].append({
                        "name": "Google Search & Scrape (Rich Content)", 
                        "articles_retrieved_count": len(scraped_articles_data),
                        "article_titles": [article.get('title', 'N/A') for article in scraped_articles_data],
                        "article_urls": [article.get('url', 'N/A') for article in scraped_articles_data]
                    })

                except Exception as e:
                    st.error(f"Error during web scraping phase with Google Search: {e}")
                    current_process_log["steps"].append({"name": "Web Scraping Error (Google)", "error": str(e)})
            
            if scraped_articles_data:
                st.success(f"Successfully processed {len(scraped_articles_data)} article(s) from the web.")
                # Create a list of URLs from the successfully processed articles for the expander title
                # This list should come from the 'url' key in each dict within scraped_articles_data
                processed_urls_titles = [(article.get('url', 'N/A'), article.get('title', 'No Title')) for article in scraped_articles_data]

                with st.expander(f"📚 View Processed Article Details ({len(processed_urls_titles)} found with content)", expanded=False):
                    for i, article_data_dict in enumerate(scraped_articles_data):
                        title = article_data_dict.get('title', 'No Title')
                        url = article_data_dict.get('url', '#')
                        domain = article_data_dict.get('domain', 'N/A')
                        extraction_method = article_data_dict.get('extraction_method', 'N/A')
                        publish_date = article_data_dict.get('publish_date')
                        extraction_note = article_data_dict.get('extraction_note', '')
                        text_snippet = (article_data_dict.get('text', '')[:200] + "...") if article_data_dict.get('text') else "No text extracted."
                        
                        st.markdown(f"**Source {i+1}: [{title}]({url})**")
                        date_display = publish_date.strftime('%Y-%m-%d') if isinstance(publish_date, datetime.datetime) else str(publish_date) if publish_date else 'N/A'
                        st.caption(f"Domain: {domain} | Published: {date_display} | Method: {extraction_method} | Note: {extraction_note}")
                        st.text(text_snippet)
                        st.markdown("---")
            else:
                st.warning("Could not retrieve significant articles from the web using Google Search. The LLM will be informed.")
            
            st.markdown("---")

            # --- Phase 3: LLM Generates Brief Information Outline ---
            st.subheader("Phase 3: Generating Brief Information Outline")
            final_outline = None 
            phase3_cot = None # Renamed from phase2_cot for clarity
            with st.spinner("LLM is analyzing retrieved content and formulating an outline..."):
                try:
                    # Pass the list of article dictionaries directly
                    # Also pass phase1_cot if your llm_interaction.py expects it for plan-guided outlining
                    final_outline, phase3_cot = llm_interaction.get_summary_and_cot( 
                        original_user_query=user_query,
                        scraped_articles_data=scraped_articles_data
                        # phase1_research_plan_cot=phase1_cot, # Uncomment if get_summary_and_cot expects this
                    )
                    st.session_state.current_outline = final_outline 
                    current_process_log["steps"].append({
                        "name": "LLM Outline Generation", 
                        "final_outline_snippet": (final_outline[:150]+"..." if final_outline else "N/A"), 
                        "cot_snippet": (phase3_cot[:150]+"..." if phase3_cot else "N/A") 
                    })
                except Exception as e:
                    st.error(f"Error in Phase 3 (LLM Outline Generation): {e}")
                    final_outline = "Could not generate an outline due to an error."
                    phase3_cot = f"An error occurred: {e}"
                    current_process_log["steps"].append({"name": "LLM Outline Generation Error", "error": str(e)})
            
            display_cot("💡 LLM's Analysis & Outline Generation (Chain of Thought)", phase3_cot)
            
            st.subheader("✅ Brief Information Outline")
            if final_outline:
                st.markdown(final_outline)
            else:
                st.error("LLM did not provide an outline.")
        
        st.session_state.history.append(current_process_log)
        st.markdown("---") 

# --- Phase 4: Optional Content Elaboration ---
if st.session_state.current_outline and \
   isinstance(st.session_state.current_outline, str) and \
   "Error:" not in st.session_state.current_outline and \
   "Outline is embedded" not in st.session_state.current_outline and \
   "No outline provided" not in st.session_state.current_outline and \
   "Could not retrieve meaningful web articles" not in st.session_state.current_outline :

    st.markdown("---") 
    if st.button("✨ Elaborate Further & Generate Detailed Content", key="elaborate_button"):
        if st.session_state.original_query_for_elaboration and st.session_state.current_outline:
            st.subheader("Phase 4: Elaborating Content")
            elaborated_text_final = None # To store the final text
            elaboration_cot = None # To store CoT for elaboration
            with st.spinner("LLM is elaborating on the outline to create detailed content..."):
                try:
                    # Assuming content_elaboration.elaborate_on_outline returns (elaborated_text, cot)
                    elaborated_text_final, elaboration_cot = content_elaboration.elaborate_on_outline(
                        original_user_query=st.session_state.original_query_for_elaboration,
                        information_outline=st.session_state.current_outline
                    )
                    
                    # Log this step
                    for log_item in reversed(st.session_state.history):
                        if log_item["query"] == st.session_state.original_query_for_elaboration:
                            log_item["steps"].append({
                                "name": "Content Elaboration",
                                "elaborated_content_snippet": (elaborated_text_final[:200]+"..." if elaborated_text_final else "N/A"),
                                "cot_snippet": (elaboration_cot[:150]+"..." if elaboration_cot else "N/A")
                            })
                            break
                except Exception as e:
                    st.error(f"Error during content elaboration: {e}")
                    elaborated_text_final = f"Error during content elaboration: {e}"
                    for log_item in reversed(st.session_state.history):
                        if log_item["query"] == st.session_state.original_query_for_elaboration:
                            log_item["steps"].append({"name": "Content Elaboration Error", "error": str(e)})
                            break
            
            display_cot("🔍 LLM's Elaboration Process (Chain of Thought)", elaboration_cot) # Display CoT for elaboration

            st.subheader("🖋️ Elaborated Content")
            if elaborated_text_final and "Error:" not in elaborated_text_final and "Failed to elaborate:" not in elaborated_text_final:
                st.markdown(elaborated_text_final)
            elif elaborated_text_final: 
                st.error(elaborated_text_final)
            else:
                st.error("Could not generate elaborated content from the LLM.")
            st.markdown("---")
        else:
            st.warning("Cannot elaborate. Original query or outline is missing from the current session.")


# --- Instructions for running ---
st.sidebar.header("How to Use")
st.sidebar.info(
    "1. Ensure your API keys (`OPENROUTER_API_KEY`, `GOOGLE_API_KEY`, `CUSTOM_SEARCH_ENGINE_ID`) "
    "are correctly set in a `.env` file in the same directory as this script, "
    "or as system environment variables.\n"
    "2. Enter your query in the text box.\n"
    "3. Click 'Get Live Answer' to get a brief outline.\n"
    "4. Optionally, click 'Elaborate Further' to expand the outline into detailed content.\n"
    "5. Observe the chain of thoughts and the final answer/content."
)
st.sidebar.header("Modules Used")
st.sidebar.markdown(
    "- `streamlit` for UI\n"
    "- `llm_interaction.py` (for CoT & outlining)\n"
    "- `google_search_scraper.py` (for Google Search & Enhanced Scraping)\n" # Updated
    "- `content_elaboration.py` (for detailed content generation)"
)

if st.sidebar.checkbox("Show Processing History (Debug)"):
    st.sidebar.subheader("Processing History")
    if not st.session_state.history:
        st.sidebar.write("No queries processed yet in this session.")
    for i, item in enumerate(reversed(st.session_state.history)): 
        with st.sidebar.expander(f"Query {len(st.session_state.history)-i}: {item['query'][:30]}...", expanded=False):
            st.json(item) 
