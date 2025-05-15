# content_elaboration.py
import os
import openai
import json
import datetime

from dotenv import load_dotenv
load_dotenv()

OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")
# Using gpt-4o-mini as requested by the user for this elaboration phase
DEFAULT_ELABORATION_MODEL_NAME = os.environ.get("ELABORATION_MODEL_NAME", "nousresearch/deephermes-3-mistral-24b-preview:free") 
OPENROUTER_API_BASE = "https://openrouter.ai/api/v1"

CLIENT = None
if OPENROUTER_API_KEY:
    CLIENT = openai.OpenAI(
        base_url=OPENROUTER_API_BASE,
        api_key=OPENROUTER_API_KEY,
    )
else:
    print("Warning: OPENROUTER_API_KEY environment variable not set. LLM calls will fail.")

CURRENT_YEAR = datetime.datetime.now().year

def _call_llm_for_elaboration(messages: list[dict], model_name: str = DEFAULT_ELABORATION_MODEL_NAME) -> str | None:
    """
    Private helper function to make a call to the LLM for content elaboration.
    Returns the elaborated content or None if an error occurs.
    """
    if not CLIENT:
        error_message = "OpenRouter client not initialized for elaboration. API key might be missing."
        print(f"Error in _call_llm_for_elaboration: {error_message}")
        return f"Error: {error_message}" 
    try:
        print(f"[Content Elaboration LLM Call] Requesting completion from model: {model_name}")
        chat_completion = CLIENT.chat.completions.create(
            model=model_name,
            messages=messages,
            temperature=0.5 # Slightly lower for more focused elaboration
        )
        
        if hasattr(chat_completion, 'error') and chat_completion.error:
            api_error_message = chat_completion.error.get('message', 'Unknown API error in response object.')
            api_error_code = chat_completion.error.get('code', 'N/A')
            error_message = (f"LLM API returned an error in the response object: "
                             f"Code {api_error_code} - {api_error_message}\n"
                             f"Full error object: {str(chat_completion.error)}")
            print(f"Error in _call_llm_for_elaboration (API error in response): {error_message}")
            return f"Error generating elaborated content: {api_error_message}"

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
            print(f"[Content Elaboration LLM Call] Received elaborated content (length: {len(response_content)}).")
            return response_content
        else:
            error_message = "LLM API call for elaboration succeeded but response structure was unexpected or content was missing.\n"
            error_message += f"Chat completion object (str): {str(chat_completion)}\n"
            print(f"Error in _call_llm_for_elaboration (unexpected structure): {error_message}")
            return "Error: Could not parse elaborated content from LLM."

    except openai.APIStatusError as e: 
        error_message = f"OpenRouter API Status Error (Status {e.status_code}): {getattr(e, 'message', str(e))}\n"
        raw_text = "N/A"
        if e.response is not None:
            try: raw_text = e.response.text
            except Exception: raw_text = "Could not read raw response text."
        error_message += f"Raw response text: {raw_text}"
        if e.status_code == 429 or "rate limit" in getattr(e, 'message', str(e)).lower():
            error_message += "\nThis appears to be a rate limit error from the API."
        print(f"Error in _call_llm_for_elaboration (APIStatusError): {error_message}")
        return f"Error generating elaborated content: API Status Error {e.status_code}"
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
        error_message = f"An unexpected error occurred during LLM call for elaboration ({type(e).__name__}): {e}."
        print(f"Error in _call_llm_for_elaboration (general exception): {error_message}")
        return f"Error generating elaborated content: {str(e)}"


def elaborate_on_outline(original_user_query: str, information_outline: str, model_name: str = DEFAULT_ELABORATION_MODEL_NAME) -> str | None:
    """
    Takes an original user query and a factual outline (with citations) and uses an LLM
    to elaborate on the points, creating well-structured content that directly addresses the original query.

    Args:
        original_user_query: The user's initial query for context and to be directly addressed.
        information_outline: The factual outline generated by a previous LLM step,
                             expected to contain bullet points and source citations.
        model_name: The LLM model to use for elaboration.

    Returns:
        A string containing the elaborated content, or None/error message if failed.
    """
    print(f"\n[Content Elaboration] Elaborating on outline for query: '{original_user_query}'")

    if not information_outline or not information_outline.strip():
        return "No outline provided to elaborate on."

    system_prompt = (
        f"You are an expert content writer and AI research analyst. Your primary task is to take a factual 'Brief Information Outline' (which includes source citations) and the original user query, then expand this outline into a well-structured, detailed, and coherent piece of content that **directly and comprehensively answers the original user query.** The current year is {CURRENT_YEAR}.\n\n"

        "Here's what you need to do:\n"
        "1.  **Understand the Goal:** The original user query was: \"{original_user_query}\". The provided outline is the factual, web-sourced basis for your response. Your elaborated content MUST serve as a direct answer to this original query.\n"
        "2.  **Source of Truth:** The 'Brief Information Outline' is your primary source of facts. You MUST NOT introduce new factual claims or data points that are not supported by this outline. Your role is to elaborate, explain, connect, and structure the existing information to answer the user's query.\n"
        "3.  **Elaborate on Outline Points:** Take each bullet point from the outline and expand on it. Provide more context, explanation, or detail where appropriate, always staying true to the information presented in the outline point and its cited source.\n"
        "4.  **Structure for Clarity:** Organize the elaborated information logically. Use clear headings (e.g., using Markdown like `## Main Topic from Query` or `### Key Aspect`) to structure the content into sections that naturally address different facets of the original user query.\n"
        "5.  **Integrate and Answer:** Weave the elaborated points from the outline into a narrative or explanation that directly and thoroughly answers the `Original User Query`. For example, if the user asked 'How can X be used for Y?', your elaborated content about X (based on the outline) should clearly explain its application to Y.\n"
        "6.  **Maintain Citations (Implicitly):** The input outline already contains citations. While elaborating, you are expanding on these cited facts. You don't need to re-cite every sentence, but the elaborated content should clearly flow from the cited outline points. You can introduce sections with phrases like 'Based on information from [Source URL mentioned in outline for relevant point]...' if it adds clarity, but the main goal is a readable narrative.\n"
        "7.  **Coherent Narrative & Flow:** Ensure the elaborated content flows well, with smooth transitions between points and sections, forming a complete answer to the user's initial question.\n"
        "8.  **Professional Tone:** Maintain a professional, informative, and helpful tone.\n"
        "9.  **Output Format:** Produce the output in well-formatted Markdown.\n\n"
        "DO NOT simply repeat the outline. Your goal is to transform the factual outline points into a comprehensive, well-explained answer to the user's original query, using only the information provided in the outline."
    )

    user_prompt_for_elaboration = (
        f"Original User Query to address: \"{original_user_query}\"\n\n"
        "Factual Brief Information Outline (with citations) to use as the basis for your answer:\n"
        "```markdown\n"
        f"{information_outline}\n"
        "```\n\n"
        "Please elaborate on the points in the outline above to create a detailed and well-structured piece of content that directly and comprehensively answers my original query. Use appropriate headings and ensure all information is based strictly on the provided outline."
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt_for_elaboration}
    ]

    elaborated_content = _call_llm_for_elaboration(messages, model_name)
    
    if elaborated_content and "Error:" in elaborated_content[:20]: # Basic check if _call_llm returned an error string
        return f"Failed to elaborate: {elaborated_content}"

    return elaborated_content

if __name__ == '__main__':
    print(f"--- Testing Content Elaboration Module (V2 - Query Focused, Current Year: {CURRENT_YEAR}) ---")
    test_model_name = os.environ.get("ELABORATION_MODEL_NAME", DEFAULT_ELABORATION_MODEL_NAME)
    print(f"Using model for elaboration tests: {test_model_name}")

    if not CLIENT:
        print("Cannot run tests: OpenRouter client not initialized. Is OPENROUTER_API_KEY set?")
    else:
        sample_original_query = "How can Model Context Protocol (MCP) be used to generate PDF reports, and what are its key features?"
        sample_outline_from_phase3 = (
            "- Model Context Protocol (MCP) is an open standard developed by Anthropic to connect AI systems with various data sources and tools. (Source: https://www.anthropic.com/news/model-context-protocol)\n"
            "- MCP allows AI agents to perform multi-step tasks autonomously, enhancing their utility beyond simple query responses. (Source: https://medium.com/@elisowski/mcp-explained-the-new-standard-connecting-ai-to-everything-79c5a1c98288)\n"
            "- The protocol simplifies integration by providing a standardized way for AI to connect with external services. (Source: https://huggingface.co/blog/Kseniase/mcp)\n"
            "- MCP supports dynamic discovery, allowing AI agents to automatically detect and utilize available MCP servers. (Source: https://huggingface.co/blog/Kseniase/mcp)\n"
            "- Developers can start using MCP by installing pre-built servers for popular tools. (Source: https://www.anthropic.com/news/model-context-protocol)"
        )

        print(f"\n--- Testing Content Elaboration ---")
        print(f"Original User Query: {sample_original_query}")
        print(f"Input Outline (from Phase 3):\n{sample_outline_from_phase3}")

        elaborated_text = elaborate_on_outline(sample_original_query, sample_outline_from_phase3, model_name=test_model_name)

        if elaborated_text:
            print(f"\n[Test Result] Elaborated Content (should address original query):\n{elaborated_text}")
        else:
            print("[Test Result] Failed to get elaborated content.")
