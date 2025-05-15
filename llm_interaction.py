# llm_interaction.py
import os
import openai
import json
import datetime # Ensure datetime is imported

from dotenv import load_dotenv
load_dotenv()

OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")
# Defaulting to model from user's provided code
DEFAULT_MODEL_NAME = os.environ.get("DEFAULT_MODEL_NAME", "nousresearch/deephermes-3-mistral-24b-preview:free") 
OPENROUTER_API_BASE = "https://openrouter.ai/api/v1"

CLIENT = None
if OPENROUTER_API_KEY:
    CLIENT = openai.OpenAI(
        base_url=OPENROUTER_API_BASE,
        api_key=OPENROUTER_API_KEY,
    )
else:
    print("Warning: OPENROUTER_API_KEY environment variable not set. LLM calls will fail.")

CURRENT_YEAR = datetime.datetime.now().year # This will be 2025 in our scenario

def _call_llm(messages: list[dict], model_name: str = DEFAULT_MODEL_NAME) -> tuple[str | None, str | None]:
    """
    Private helper function to make a call to the LLM.
    Includes robust error handling and response validation.
    """
    if not CLIENT:
        error_message = "OpenRouter client not initialized. API key might be missing."
        print(f"Error in _call_llm: {error_message}")
        return None, error_message
    try:
        print(f"[LLM Call] Requesting completion from model: {model_name} with {len(messages)} messages.")
        chat_completion = CLIENT.chat.completions.create(
            model=model_name,
            messages=messages,
            temperature=0.2 
        )
        
        if hasattr(chat_completion, 'error') and chat_completion.error:
            api_error_message = chat_completion.error.get('message', 'Unknown API error in response object.')
            api_error_code = chat_completion.error.get('code', 'N/A')
            error_message = (f"LLM API returned an error in the response object: "
                             f"Code {api_error_code} - {api_error_message}\n"
                             f"Full error object: {str(chat_completion.error)}")
            print(f"Error in _call_llm (API error in response): {error_message}")
            return None, error_message

        if chat_completion and \
           hasattr(chat_completion, 'choices') and \
           chat_completion.choices and \
           len(chat_completion.choices) > 0 and \
           chat_completion.choices[0] and \
           hasattr(chat_completion.choices[0], 'message') and \
           chat_completion.choices[0].message and \
           hasattr(chat_completion.choices[0].message, 'content') and \
           chat_completion.choices[0].message.content is not None:
            response_content = chat_completion.choices[0].message.content
            print(f"[LLM Call] Received response content (length: {len(response_content)}).")
            return response_content, response_content 
        else:
            error_message = "LLM API call succeeded but response structure was unexpected or content was missing.\n"
            error_message += f"Chat completion object (str): {str(chat_completion)}\n"
            print(f"Error in _call_llm (unexpected structure): {error_message}")
            return None, error_message
    except openai.APIStatusError as e: 
        error_message = f"OpenRouter API Status Error (Status {e.status_code}): {getattr(e, 'message', str(e))}\n"
        raw_text = "N/A"
        if e.response is not None:
            try: raw_text = e.response.text
            except Exception: raw_text = "Could not read raw response text."
        error_message += f"Raw response text: {raw_text}"
        if e.status_code == 429 or "rate limit" in getattr(e, 'message', str(e)).lower():
            error_message += "\nThis appears to be a rate limit error from the API."
        print(f"Error in _call_llm (APIStatusError): {error_message}")
        return None, error_message
    except openai.APIConnectionError as e:
        error_message = f"OpenRouter API Connection Error: {e}. Network issue or server unavailable."
    except openai.RateLimitError as e: 
        error_message = f"OpenRouter Rate Limit Exceeded (RateLimitError exception): {e}"
    except openai.AuthenticationError as e: 
        error_message = f"OpenRouter Authentication Error: {e}. Check API key."
    except openai.APIError as e: 
        error_message = f"OpenRouter APIError ({type(e).__name__}): {e}."
        raw_text = "N/A"
        if hasattr(e, 'response') and e.response is not None and hasattr(e.response, 'text'):
            try: raw_text = e.response.text
            except Exception: raw_text = "Could not read raw response text."
        error_message += f" Raw response: {raw_text}"
    except json.JSONDecodeError as e: 
        error_message = f"JSON Decode Error during LLM call ({type(e).__name__}): {e}.\n"
        error_message += ("API server returned non-JSON response. Possible server issue or HTML error page.")
    except Exception as e: 
        error_message = f"An truly unexpected error occurred during LLM call ({type(e).__name__}): {e}."
        print(f"Error in _call_llm (general exception): {error_message}")
        return None, error_message


def get_search_query_and_cot(user_query: str, model_name: str = DEFAULT_MODEL_NAME) -> tuple[str | None, str | None]:
    """
    Phase 1: Gets the refined search query and the LLM's chain of thought for planning.
    This version uses the prompt structure from the user's provided code.
    """
    print(f"\n[LLM Interaction] Phase 1: Getting search query for: '{user_query}'")
    
    system_prompt = (
        f"You are a search query generation assistant. The current year is {CURRENT_YEAR}. "
        "Your task is to meticulously think step-by-step, like a detective, to generate the best possible search engine query for the user's request. "
        "You MUST provide your response in a specific structure. \n\n"
        "First, as a deep thinking AI, show your detailed internal monologue and deliberations. You may use extremely long chains of thought to deeply consider the problem and deliberate with yourself via systematic reasoning processes to help come to a correct solution. This is your free-form thinking space.\n\n"
        
        "After your internal monologue, you MUST then present your structured response in EXACTLY two parts, with NO other text whatsoever between or after these parts:\n\n"
        
        "PART 1: DETAILED CHAIN OF THOUGHT\n"
        "Prefix this part with 'Chain of thought:'.\n"
        "In this section, formally elaborate on your reasoning process. Your steps should include:\n"
        "  a. Initial understanding: What is the user's core information need based on their query: '{user_query}'?\n"
        "  b. Keyword identification: What are the most crucial keywords and concepts? Are there synonyms, related terms, or domain-specific jargon to consider?\n"
        "  c. Timeframe considerations: Does the query mention specific years (e.g., '{CURRENT_YEAR}', '2025', 'last year') or imply a need for recent information ('latest', 'current')? These timeframes MUST be preserved AS IS in the final search query.\n"
        "  d. Query construction (Tree of Thought): Generate at least 3 distinct search query options that would potentially answer the user's need. For each option:\n"
        "    - Explain the rationale for why it could be effective\n"
        "    - Score it on three dimensions using a 1–10 scale: Precision (relevance), Recall (coverage), and Recency (freshness)\n"
        "    - Discuss the pros and cons\n"
        "  Then, reason about which query is optimal based on scores and scope. Prune the weaker candidates and explain the final selection.\n"
        "  e. Final keyword selection: What exact keyword phrase will form the final search query?\n\n"
        
        "PART 2: SEARCH QUERY\n"
        "Prefix this part with 'SEARCH_QUERY:'.\n"
        "This line must contain ONLY the final, concise, and effective search engine query derived from your chain of thought above. "
        "Do NOT add any text, explanation, markdown, or disclaimers after the SEARCH_QUERY line. Your response must end immediately after the search query itself.\n\n"
        
        "--- STRICT FORMAT EXAMPLE (User query: 'latest AI news 2025') ---\n"
        "INTERNAL MONOLOGUE / DELIBERATIONS:\n" 
        "The user wants the most recent news about Artificial Intelligence specifically for the year 2025. I need to ensure the query is targeted to this year and the concept of 'latest news'. Keywords: AI, news, 2025, latest. Option 1: 'latest AI news 2025' - Direct and good. Option 2: 'AI 2025 developments' - Broader, might miss news. Option 1 seems best.\n\n"
        "Chain of thought:\n"
        "a. Initial understanding: The user wants the most recent news about Artificial Intelligence specifically for the year 2025.\n"
        "b. Keyword identification: 'AI', 'news', '2025'. 'Latest' implies recency.\n"
        "c. Timeframe considerations: The year '2025' is explicitly mentioned and must be used.\n"
        "d. Query construction:\n"
        "   Option 1: 'latest AI news 2025' — High precision, high recall, good recency. Scores: Precision 9/10, Recall 9/10, Recency 9/10\n"
        "   Option 2: 'AI breakthroughs 2025' — Focuses more on innovation but might miss general news. Precision 8/10, Recall 7/10, Recency 8/10\n"
        "   Option 3: 'AI technology updates 2025' — Could pull technical and industry info but may lack major headline news. Precision 7/10, Recall 8/10, Recency 7/10\n"
        "   Final evaluation: Option 1 is the most balanced and directly aligned with the user's need for current news.\n"
        "e. Final keyword selection: latest AI news 2025\n"
        "SEARCH_QUERY: latest AI news 2025\n"
        "--- END OF EXAMPLE ---"
    )
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_query}
    ]

    full_response, error_message_from_call = _call_llm(messages, model_name)

    if error_message_from_call and not full_response: 
        return None, error_message_from_call

    if full_response:
        lines = full_response.strip().split('\n')
        search_query = None
        
        for line_idx, line_content in enumerate(lines): 
            if line_content.startswith("SEARCH_QUERY:"):
                potential_query = line_content.replace("SEARCH_QUERY:", "").strip()
                if potential_query: 
                    search_query = potential_query
                    if line_idx < len(lines) - 1 and lines[line_idx+1].strip():
                        print(f"[LLM Interaction Warning] Text found after SEARCH_QUERY line: '{lines[line_idx+1]}...'")
                    break 
        
        phase_1_chain_of_thought = full_response 
        
        if not search_query:
             print("[LLM Interaction Warning] 'SEARCH_QUERY:' prefix not found or query was empty in Phase 1 LLM response. Cannot determine search query. The full response was:")
             print(full_response) 
             return None, phase_1_chain_of_thought

        print(f"[LLM Interaction] Suggested Search Query: {search_query}")
        return search_query, phase_1_chain_of_thought
    
    return None, error_message_from_call if error_message_from_call else "LLM did not return a response for Phase 1."


def get_summary_and_cot(original_user_query: str, scraped_articles_data: list[dict], model_name: str = DEFAULT_MODEL_NAME) -> tuple[str | None, str | None]:
    """
    Phase 2: Gets a brief outline/digest based on a list of scraped article dictionaries.
    Each dictionary in scraped_articles_data should have 'url', 'title', 'text', 'publish_date'.
    This version instructs the LLM for an outline format with numbered source citations and a References section,
    and encourages a preceding internal monologue.
    """
    print(f"\n[LLM Interaction] Phase 2: Generating outline for query: '{original_user_query}'")

    meaningful_scraped_articles = [
        article for article in scraped_articles_data 
        if article and isinstance(article, dict) and article.get('text') and article.get('text').strip()
    ]

    if not meaningful_scraped_articles:
        system_prompt_no_content = (
            f"You are an expert information analyst. The current year is {CURRENT_YEAR}. "
            "You were asked to provide an outline for the query: "
            f"'{original_user_query}'. However, no meaningful web content (articles with text) was provided after attempting a web search. "
            "Your task is to state that you could not find specific web articles to create an outline and therefore cannot provide a web-based response. "
            "Do not use your general knowledge. Your entire response should be ONLY this statement, prefixed with 'BRIEF_INFORMATION_OUTLINE:'."
        )
        messages_no_content = [
            {"role": "system", "content": system_prompt_no_content}
        ]
        statement, error_no_content = _call_llm(messages_no_content, model_name)
        if statement: 
            parsed_statement = statement.replace("BRIEF_INFORMATION_OUTLINE:", "").strip() if statement.startswith("BRIEF_INFORMATION_OUTLINE:") else statement
            return parsed_statement, statement 
        else:
            return "Could not retrieve meaningful web articles, and an error occurred generating a statement.", error_no_content


    formatted_scraped_content_for_llm = ""
    total_chars_processed = 0
    max_total_chars_for_llm_context = 60000 
    
    source_url_mapping_for_prompt = [] 
    for i, article in enumerate(meaningful_scraped_articles):
        url = article.get('url', 'N/A')
        source_url_mapping_for_prompt.append(f"[{i+1}] {url}") 

        title = article.get('title', 'N/A')
        
        # --- CORRECTED DATE HANDLING ---
        publish_date_obj = article.get('publish_date') # This should be a datetime object or None
        if publish_date_obj and isinstance(publish_date_obj, datetime.datetime):
            date_info = f"Published: {publish_date_obj.strftime('%Y-%m-%d')}"
        elif isinstance(publish_date_obj, str): # If it's already a string for some reason
            date_info = f"Published: {publish_date_obj.split('T')[0] if 'T' in publish_date_obj else publish_date_obj}"
        else:
            date_info = "Published: N/A"
        # --- END OF CORRECTION ---
            
        text_content = article.get('text', '')
        
        source_tag = (
           f"--- Source [{i+1}] ---\n" 
           f"URL: {url}\n" 
           f"Title: {title}\n"
           f"{date_info}\n"
           f"(Consider this source independently when reasoning. Treat each article as a separate evidence chunk.)\n"
           "Content:\n"
        )
        
        available_chars_for_this_article = max_total_chars_for_llm_context - total_chars_processed - len(source_tag) - 20 
        if available_chars_for_this_article <= 50: 
            break 

        formatted_scraped_content_for_llm += source_tag
        formatted_scraped_content_for_llm += text_content[:available_chars_for_this_article]
        formatted_scraped_content_for_llm += f"\n--- End of Source [{i+1}] ---\n\n" 
        total_chars_processed += len(source_tag) + len(text_content[:available_chars_for_this_article]) + 20

    numbered_source_list_for_prompt = "\n".join(source_url_mapping_for_prompt)

    system_prompt_with_content = (
        f"You are a highly advanced AI news and research analyst. Your task is to create a 'Brief Information Outline' or 'Key Findings Digest' from provided web content. "
        f"The current year is {CURRENT_YEAR}. You are given:\n"
        f"- A user query: '{original_user_query}'\n"
        f"- Web articles scraped from multiple sources. Each article is presented below as 'Source [number]' and includes its URL, Title, Publish Date, and Content.\n"
        f"For your reference, here is the mapping of source numbers to their full URLs that you will use for citations:\n"
        f"REFERENCE_LIST_START\n{numbered_source_list_for_prompt}\nREFERENCE_LIST_END\n\n"
        "Your job is to perform multi-step reasoning over the content to produce this outline. "
        "Use advanced reasoning, like an expert model, to identify the most salient facts and present them clearly.\n\n"
        "⚠️ CRITICAL: Do not add any information not explicitly mentioned in the provided sources. Do not hallucinate. Be evidence-driven.\n\n"
        "IMPORTANT: Even if an article lacks a publish date or contains general information, you must still extract and include any potentially relevant news, events, or updates that could be useful for the user's query. Prioritize event-based or specific snippets over generic descriptions, but do not discard articles solely due to missing dates or lack of explicit recency. If recent news is scarce, include all possible relevant facts from the available articles.\n\n"
        "You MUST provide your response in a specific structure. \n\n"
        "First, as a deep thinking AI, show your detailed internal monologue and deliberations. You may use extremely long chains of thought to deeply consider the problem, evaluate each source, and plan your outline. This is your free-form thinking space before the structured parts.\n\n"
        "After your internal monologue, you MUST then present your structured response in EXACTLY two parts, with NO other text whatsoever between or after these parts:\n\n"
        "PART 1: Chain of thought:\n"
        "Prefix this part with 'Chain of thought:'.\n"
        "- For each source (e.g., Source [1], Source [2]), reason through its:\n"
        "  a. Relevance to the query: '{original_user_query}'\n"
        "  b. Timeliness (using its Published Date and content cues, but do not discard if missing)\n"
        "  c. Credibility and clarity of the information.\n"
        "  d. Key facts, figures, or distinct pieces of information extracted that directly help answer the query or form part of the outline. When referring to a source in your thought process, use its number (e.g., 'Source [1] states...').\n"
        "- Then provide a synthesis plan explaining how you'll organize these key points into a coherent outline, prioritizing the most relevant and timely information. Indicate which source numbers support each planned point.\n\n"
        "PART 2: BRIEF_INFORMATION_OUTLINE:\n"
        "Prefix this part with 'BRIEF_INFORMATION_OUTLINE:'.\n"
        "- Present the key findings as a structured, grouped news digest, using clear section headers (with relevant emojis) for each major news topic.\n"
        "- Under each section header, use concise bullet points for distinct news items or facts.\n"
        "- Each bullet point should directly state a key piece of information from the sources.\n"
        "- After each bullet point, cite the source(s) using their corresponding number(s) in square brackets, like [1] or [1, 2].\n"
        "- If multiple articles cover the same event, group them under the same section and cite all relevant sources.\n"
        "- Prioritize recency and importance. If information is conflicting, you may note it briefly.\n"
        "- IMPORTANT: Extract and include ALL distinct newsworthy events, facts, or updates found in the sources, even if only mentioned in a single article. Do NOT omit relevant items just because they appear in only one source.\n"
        "- If a news item is only present in one source, still include it as a separate bullet point with its citation.\n"
        "- Use a style similar to this example:\n"
        "BRIEF_INFORMATION_OUTLINE:\n"
        "🚌 Tragic Bus Fire Claims Five Lives\n"
        "- A private sleeper bus traveling from Begusarai to Delhi caught fire on Kisan Path in Lucknow, resulting in five deaths, including two children. The driver fled the scene and is being sought by authorities. [1, 2, 3]\n"
        "\n🦁 Zoos Closed Amid Bird Flu Outbreak\n"
        "- Following the death of a tigress due to bird flu at Gorakhpur Zoo, zoos in Lucknow, Kanpur, and Gorakhpur have been temporarily closed. Carnivorous animals are being fed mutton instead of poultry. [4, 5]\n"
        "\n🛡️ BrahMos Missile Facility Inaugurated\n"
        "- Defence Minister inaugurated a new BrahMos missile manufacturing plant in Lucknow, set to produce up to 100 supersonic cruise missiles annually. [6, 7]\n"
        "\n🍲 Bada Mangal Festival Commences\n"
        "- The annual Bada Mangal festival has begun in Lucknow, with over 350 community feasts registered. [8, 9]\n"
        "\n🧪 Crackdown on Food Adulteration\n"
        "- The Chief Minister announced strict measures against food and medicine adulteration, including public shaming and new testing labs. [10]\n"
        "\nReferences:\n"
        "[1] https://example.com/news1\n[2] https://example.com/news2\n...\n"
        "--- END OF EXAMPLE ---\n"
        "If no sources are relevant or timely, state that clearly in the BRIEF_INFORMATION_OUTLINE.\n"
        "**AFTER the grouped outline, create a 'References:' section.**\n"
        "Under 'References:', list each source number you cited in the outline, followed by its full URL (taken from the REFERENCE_LIST_START/END block provided to you).\n"
        "Ensure NO text appears after the References section.\n"
    )
    
    user_prompt_for_summary = (
        f"Original User Query: \"{original_user_query}\"\n\n"
        f"Below is web article content with metadata. Analyze it and produce a BRIEF_INFORMATION_OUTLINE in the specified format, ensuring to use numbered citations and include a References section as instructed.\n\n"
        f"{formatted_scraped_content_for_llm}"
    )

    messages = [
        {"role": "system", "content": system_prompt_with_content},
        {"role": "user", "content": user_prompt_for_summary} 
    ]

    full_response, error_message_from_call = _call_llm(messages, model_name)

    if error_message_from_call and not full_response:
        return None, error_message_from_call
    
    if full_response:
        phase_2_chain_of_thought = full_response 
        final_outline = None 

        if "BRIEF_INFORMATION_OUTLINE:" in full_response:
            parts = full_response.split("BRIEF_INFORMATION_OUTLINE:", 1)
            if len(parts) > 1:
                final_outline = parts[1].strip()
            else: 
                final_outline = "Outline was expected after 'BRIEF_INFORMATION_OUTLINE:' tag, but none was found."
        else:
            print("[LLM Interaction Warning] 'BRIEF_INFORMATION_OUTLINE:' prefix not found in Phase 2. The full response will be treated as CoT, and outline might be missing or embedded.")
            final_outline = "Outline is embedded within the Chain of Thought above. 'BRIEF_INFORMATION_OUTLINE:' tag was not found."

        return final_outline, phase_2_chain_of_thought 

    return None, error_message_from_call if error_message_from_call else "LLM did not return a response for Phase 2."


if __name__ == '__main__':
    print(f"--- Testing LLM Interaction Module (User's Prompt Style + Date Fix, Current Year: {CURRENT_YEAR}) ---")
    test_model_name = os.environ.get("TEST_LLM_MODEL", DEFAULT_MODEL_NAME)
    print(f"Using model for tests: {test_model_name}")

    if not CLIENT:
        print("Cannot run tests: OpenRouter client not initialized. Is OPENROUTER_API_KEY set?")
    else:
        # Test Phase 1 
        test_query_p1 = f"ipl {CURRENT_YEAR} current situation" 
        print(f"\n--- Testing Phase 1 with query: '{test_query_p1}' ---")
        search_q, p1_cot = get_search_query_and_cot(test_query_p1, model_name=test_model_name)
        if search_q: 
            print(f"[Test Result P1] Search Query: {search_q}")
        if p1_cot: 
            print(f"[Test Result P1] CoT (Full Response from LLM):\n{p1_cot}")
        
        # Test Phase 2 (with structured article data for numbered citation)
        test_query_p2_citation = f"Key AI developments in {CURRENT_YEAR}"
        sample_articles_p2_citation = [
            {
                'url': 'https://example.com/ai_dev_1', 
                'title': f'New AI Chips Boost Performance in {CURRENT_YEAR}', 
                'text': f'Several companies announced new AI accelerator chips in early {CURRENT_YEAR}. These chips promise up to 50% performance increase for training large models. Key players include Nvidia and Intel.', 
                'publish_date': datetime.datetime(CURRENT_YEAR, 3, 15, 10, 0, 0) # Actual datetime object
            },
            {
                'url': 'https://example.com/ai_ethics_report', 
                'title': f'{CURRENT_YEAR} Report on AI Ethics and Regulation', 
                'text': f'A major report published in April {CURRENT_YEAR} highlighted the growing need for AI regulation. It called for international cooperation on ethical guidelines. The report also noted public concerns over AI bias.', 
                'publish_date': f'{CURRENT_YEAR}-04-20T14:30:00Z' # String, to test mixed types
            },
            {
                'url': 'https://example.com/no_date_news',
                'title': 'Timeless AI Insights',
                'text': 'AI continues to evolve rapidly.',
                'publish_date': None # No date
            }
        ]
        print(f"\n--- Testing Phase 2 with query: '{test_query_p2_citation}' (for numbered citation format) ---")
        outline, p2_cot_outline = get_summary_and_cot(test_query_p2_citation, sample_articles_p2_citation, model_name=test_model_name)
        if outline:
            print(f"[Test Result P2 Numbered Citation] Parsed Outline & References:\n{outline}")
        if p2_cot_outline:
            print(f"[Test Result P2 Numbered Citation] CoT (Full Response from LLM):\n{p2_cot_outline}")
