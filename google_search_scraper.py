# google_search_scraper.py
import os
import time
import json
import random
from urllib.parse import urlparse, quote_plus
from datetime import datetime, timedelta, timezone 
from dotenv import load_dotenv
from newspaper import Article, ArticleException, Config as NewspaperConfig
from bs4 import BeautifulSoup
import logging
import requests
import re
from collections import Counter

# Google API Client
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

# Custom extractors dictionary (empty by default)
CUSTOM_EXTRACTORS = {}

# Selenium (Optional)
SELENIUM_ENABLED = os.getenv('SELENIUM_ENABLED', 'true').lower() == 'true'
SELENIUM_DRIVER_PATH = os.getenv('SELENIUM_DRIVER_PATH') 
USER_AGENT_LIST = [ # Define USER_AGENT_LIST before it's used by SELENIUM_OPTIONS
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Safari/605.1.15',
    'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:121.0) Gecko/20100101 Firefox/121.0'
]

if SELENIUM_ENABLED:
    try:
        from selenium import webdriver
        from selenium.webdriver.chrome.options import Options as ChromeOptions
        from selenium.webdriver.chrome.service import Service as ChromeService
        from selenium.common.exceptions import WebDriverException
        CHROME_OPTIONS = ChromeOptions()
        CHROME_OPTIONS.add_argument('--headless')
        CHROME_OPTIONS.add_argument('--disable-gpu')
        CHROME_OPTIONS.add_argument('--no-sandbox') 
        CHROME_OPTIONS.add_argument('--disable-dev-shm-usage') 
        CHROME_OPTIONS.add_argument(f'user-agent={random.choice(USER_AGENT_LIST)}') 
        CHROME_OPTIONS.add_experimental_option('excludeSwitches', ['enable-logging'])
    except ImportError:
        logging.warning("Selenium library not installed, but SELENIUM_ENABLED is true. Disabling Selenium.")
        SELENIUM_ENABLED = False


# Set up logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('google_search_scraper')

# Load environment variables
load_dotenv()

# Configuration
RESULTS_PER_API_CALL = int(os.getenv('RESULTS_PER_API_CALL_GOOGLE', 5)) 
TOTAL_URLS_TO_PROCESS_LIMIT = int(os.getenv('TOTAL_URLS_TO_PROCESS_LIMIT_GOOGLE', 10)) 
MIN_CONTENT_LENGTH = int(os.getenv('MIN_CONTENT_LENGTH', 250)) 
MAX_API_RETRIES = 2 
BACKOFF_FACTOR_API = 3 
MAX_HTML_RETRIES = 2 
BACKOFF_FACTOR_HTML = 1

TRUSTED_DOMAINS_GENERAL = os.getenv(
    'TRUSTED_DOMAINS_GENERAL',
    'wikipedia.org,reuters.com,apnews.com,bbc.com,nytimes.com,wsj.com,forbes.com,bloomberg.com,techcrunch.com,theverge.com,wired.com,arstechnica.com,instagram.com,facebook.com,twitter.com,linkedin.com,reddit.com,medium.com,github.com,stackoverflow.com,youtube.com'
).split(',') 
TRUSTED_DOMAINS_IPL = os.getenv(
    'TRUSTED_DOMAINS_IPL',
    'espncricinfo.com,cricbuzz.com,indiatimes.com,timesofindia.indiatimes.com,'
    'ndtv.com/sport,hindustantimes.com/cricket,bcci.tv,cricketworld.com,'
    'sportskeeda.com/cricket,news18.com/cricket,indianexpress.com/sports/cricket,thehindu.com/sport/cricket,sportstar.thehindu.com,iplt20.com'
).split(',')
TRUSTED_DOMAINS_LUCKNOW = os.getenv(
    'TRUSTED_DOMAINS_LUCKNOW',
    'timesofindia.indiatimes.com,cities.hindustantimes.com,indianexpress.com,amarujala.com,jagran.com,news18.com,ndtv.com,thehindu.com,dainikbhaskar.com,patrika.com,allevents.in,bookmyshow.com'
).split(',') 


# Google API Configuration
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
CUSTOM_SEARCH_ENGINE_ID = os.getenv("CUSTOM_SEARCH_ENGINE_ID")

# newspaper3k Configuration
NP_CONFIG = NewspaperConfig()
NP_CONFIG.browser_user_agent = random.choice(USER_AGENT_LIST)
NP_CONFIG.request_timeout = 20
NP_CONFIG.fetch_images = False
NP_CONFIG.memoize_articles = False
NP_CONFIG.verbose = False 


def get_random_user_agent():
    return random.choice(USER_AGENT_LIST)

def get_html_with_headers(url: str) -> str:
    logger.info(f"[HTML Fetch] Attempting to fetch: {url}")
    html_content = ''
    session = requests.Session()
    headers = {
        'User-Agent': get_random_user_agent(),
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.5', 'Referer': 'https://www.google.com/',
        'DNT': '1', 'Upgrade-Insecure-Requests': '1'
    }
    try:
        for attempt in range(MAX_HTML_RETRIES):
            try:
                resp = session.get(url, headers=headers, timeout=15) 
                resp.raise_for_status()
                html_content = resp.text
                logger.info(f"  [HTML Fetch] Successfully fetched with requests from {url}")
                return html_content
            except requests.exceptions.RequestException as e:
                logger.warning(f"  [HTML Fetch] Requests attempt {attempt+1} failed for {url}: {e}")
                if attempt < MAX_HTML_RETRIES - 1:
                    time.sleep(BACKOFF_FACTOR_HTML * (2**attempt))
                else: 
                    logger.error(f"  [HTML Fetch] Requests failed after {MAX_HTML_RETRIES} attempts for {url}.")
    except Exception as e: 
        logger.error(f"  [HTML Fetch] Unexpected error during requests for {url}: {e}")

    if not html_content and SELENIUM_ENABLED:
        logger.info(f"  [HTML Fetch] Falling back to Selenium for {url}")
        driver = None # Initialize driver to None for the finally block
        try:
            if SELENIUM_DRIVER_PATH:
                service = ChromeService(executable_path=SELENIUM_DRIVER_PATH)
                driver = webdriver.Chrome(service=service, options=CHROME_OPTIONS)
            else: 
                driver = webdriver.Chrome(options=CHROME_OPTIONS)
            
            driver.set_page_load_timeout(30) 
            driver.get(url)
            time.sleep(random.uniform(3,5)) 
            html_content = driver.page_source
            logger.info(f"  [HTML Fetch] Successfully fetched with Selenium from {url} (Length: {len(html_content)})")
        except WebDriverException as e:
            logger.error(f"  [HTML Fetch] Selenium WebDriverException for {url}: {e}")
        except Exception as e: 
            logger.error(f"  [HTML Fetch] Unexpected Selenium error for {url}: {e}")
        finally:
            if driver is not None: # Check if driver was initialized
                driver.quit()
    
    if not html_content:
        logger.error(f"  [HTML Fetch] Failed to fetch HTML content for {url} after all attempts.")
    return html_content


def determine_query_params_for_google(query: str, location: str = None, lookback_hours: int = None) -> tuple[str, list[str], int, str | None]:
    query_lower = query.lower()
    date_restrict_api = None
    trusted_domains_for_query = list(TRUSTED_DOMAINS_GENERAL) 
    num_api_results = RESULTS_PER_API_CALL 

    query_for_api = query 
    if location:
        location_lower = location.lower()
        if location_lower not in query_lower:
            if "news" in query_lower or "latest" in query_lower or "updates" in query_lower or "current" in query_lower:
                query_for_api = f"{location} {query}" 
            else:
                query_for_api = f"{query} in {location}"
        logger.info(f"Location context: Using query '{query_for_api}'")
        if "lucknow" in location_lower:
             trusted_domains_for_query.extend(TRUSTED_DOMAINS_LUCKNOW)


    if lookback_hours:
        if lookback_hours <= 24: date_restrict_api = "d1" 
        elif lookback_hours <= 24 * 7: date_restrict_api = "w1" 
        elif lookback_hours <= 24 * 30: date_restrict_api = "m1" 
    elif "last 24 hours" in query_lower or "today" in query_lower:
        date_restrict_api = "d1"
    elif "last week" in query_lower or "past week" in query_lower:
        date_restrict_api = "w1"
    elif "last month" in query_lower or "past month" in query_lower:
        date_restrict_api = "m1"
    elif "news" in query_lower or "latest" in query_lower or "current" in query_lower:
        logger.info("General news/latest query detected, setting dateRestrict to past 7 days (d7).")
        date_restrict_api = "d7" 
    
    current_year_str = str(datetime.now().year) 
    if current_year_str in query and not (date_restrict_api or "latest" in query_lower or "current" in query_lower):
        logger.info(f"Query contains specific year {current_year_str} and no recency term. Removing dateRestrict.")
        date_restrict_api = None 
    
    if "ipl" in query_lower or "cricket" in query_lower:
        trusted_domains_for_query.extend(TRUSTED_DOMAINS_IPL)

    if date_restrict_api or "news" in query_lower or "latest" in query_lower or "updates" in query_lower:
        num_api_results = min(max(RESULTS_PER_API_CALL, 7), 10) 
        logger.info(f"News context detected for Google. Requesting up to: {num_api_results} results per API query.")

    original_query_parts = query.split() 
    if len(original_query_parts) > 2 and not (query.startswith('"') and query.endswith('"')):
        if sum(1 for word in original_query_parts if word and word[0].isupper()) > len(original_query_parts) / 2:
            quoted_entity = f'"{query}"'
            query_for_api = f"{quoted_entity} in {location}" if location and location_lower not in quoted_entity.lower() else quoted_entity
            logger.info(f"Specific entity heuristic applied. Modified query for Google API: {query_for_api}")
    
    return query_for_api, list(set(trusted_domains_for_query)), num_api_results, date_restrict_api


def build_google_search_queries(base_query_for_api: str, is_entity_search: bool, trusted_domains_for_this_query: list[str]) -> list[str]:
    queries = set()
    queries.add(base_query_for_api) 

    core_query = base_query_for_api
    if is_entity_search: 
        if base_query_for_api.startswith('"') and base_query_for_api.endswith('"'):
            core_query = base_query_for_api.strip('"')
            queries.add(core_query) 
            # If location was part of base_query_for_api (e.g. "Entity" in Location), try entity alone
            if " in " in core_query:
                queries.add(core_query.split(" in ")[0].strip('"'))


    if not ("news" in core_query.lower() or "updates" in core_query.lower() or "latest" in core_query.lower()):
        queries.add(f"{core_query} latest news")
        queries.add(f"{core_query} updates")

    if trusted_domains_for_this_query:
        random.shuffle(trusted_domains_for_this_query) 
        for i, domain in enumerate(trusted_domains_for_this_query):
            if i >= 2 and not is_entity_search : break # Fewer site-specific for general
            if i >= 1 and is_entity_search: break # Very few for specific entities
            queries.add(f"{core_query} site:{domain.strip()}") # Use core_query for site search
            
    final_queries = list(queries)
    if base_query_for_api in final_queries: 
        final_queries.remove(base_query_for_api)
        final_queries.insert(0, base_query_for_api)
        
    logger.info(f"[Query Builder] Generated {len(final_queries)} refined queries for Google.")
    return final_queries[:5] 


def get_urls_from_google_api(query: str, num_results: int, date_restrict_param: str | None) -> list[dict]:
    if not GOOGLE_API_KEY or not CUSTOM_SEARCH_ENGINE_ID:
        logger.error("GOOGLE_API_KEY or CUSTOM_SEARCH_ENGINE_ID not set.")
        return []

    urls_info_list = []
    is_news_query_flag = "news" in query.lower() or "latest" in query.lower() or date_restrict_param is not None

    try:
        logger.info(f"  [Google API] Sending query: '{query}' (Requesting {num_results} results)")
        service = build("customsearch", "v1", developerKey=GOOGLE_API_KEY, cache_discovery=False)
        
        request_params = {'q': query, 'cx': CUSTOM_SEARCH_ENGINE_ID, 'num': min(num_results, 10)}
        if date_restrict_param:
            request_params['dateRestrict'] = date_restrict_param
            logger.info(f"  [Google API] Applying dateRestrict: {date_restrict_param} for query: '{query}'")

        res = service.cse().list(**request_params).execute()

        if 'items' in res:
            for item in res['items']:
                url = item.get('link')
                title = item.get('title', 'N/A')
                snippet = item.get('snippet', '') 
                if not url: continue
                
                parsed_u = urlparse(url)
                domain = parsed_u.netloc

                if is_news_query_flag and domain in LOW_QUALITY_DOMAINS_FOR_NEWS:
                    logger.info(f"    https://officialfilter.com/ Skipping low-quality/social media news source: {url}")
                    continue

                if parsed_u.scheme not in ['http', 'https']: continue
                if any(ext in url.lower() for ext in ['.pdf', '.doc', '.xls', '.ppt', '.zip', '.exe', '.jpg', '.png', '.gif']):
                    logger.info(f"    https://officialfilter.com/ Skipping direct file/image link: {url}")
                    continue
                
                # Heuristic for homepages (User's Step 2 suggestion)
                is_likely_article = "articleshow" in url or "/news/" in url or f"/{str(datetime.now().year)}/" in url or f"/{str(datetime.now().year-1)}/" in url or len(parsed_u.path.split('/')) > 3
                # Relax article heuristic for trusted Lucknow news domains
                trusted_lucknow_domains = [d.split('/')[0].lower() for d in TRUSTED_DOMAINS_LUCKNOW]
                # Always treat Amar Ujala, Jagran, Dainik Bhaskar as likely articles for news queries
                hindi_news_domains = ["amarujala.com", "jagran.com", "dainikbhaskar.com"]
                if domain.lower() in trusted_lucknow_domains or domain.lower() in hindi_news_domains:
                    is_likely_article = True
                if not is_likely_article:
                    # For news queries, be stricter with homepages
                    if is_news_query_flag and (parsed_u.path in ['/', ''] or len(parsed_u.path.split('/')) <=2) :
                         if not any(kw in snippet.lower() for kw in query.lower().split() if len(kw)>3): 
                            logger.info(f"    https://officialfilter.com/ Skipping likely homepage with less relevant snippet for news query: {url}")
                            continue
                    elif not is_news_query_flag: # Less strict for non-news
                        pass 
                    else: 
                        logger.info(f"    https://officialfilter.com/ Skipping URL by path/article heuristic: {url}")
                        continue
                
                urls_info_list.append({'url': url, 'title': title, 'domain': domain, 'snippet': snippet})
        else:
            logger.warning(f"  [Google API] No 'items' in Google API response for query: '{query}'.")
            
    except HttpError as e:
        logger.error(f"  [Google API] HttpError for query '{query}': {e.resp.status} {e._get_reason()}")
        if e.resp.status == 429 or e.resp.status == 403: 
            logger.warning(f"  [Google API] Rate limit or quota likely exceeded (Status: {e.resp.status}).")
    except Exception as e:
        logger.error(f"  [Google API] Unexpected error for query '{query}': {e}")
    
    logger.info(f"  [Google API] Retrieved {len(urls_info_list)} valid URLs for query: '{query}'")
    return urls_info_list

LOW_QUALITY_DOMAINS_FOR_NEWS = [
    "indiatoday.in", "news18.com", "timesnownews.com", "republicworld.com", "opindia.com", "oneindia.com",
    "zeenews.india.com", "firstpost.com", "wionews.com", "dnaindia.com", "newsbytesapp.com", "thequint.com",
    "swarajyamag.com", "abplive.com", "ibtimes.co.in", "latestly.com", "theprint.in", "thewire.in", "scroll.in",
    "livemint.com", "deccanherald.com", "freepressjournal.in", "financialexpress.com", "business-standard.com",
    "newsx.com", "theweek.in", "outlookindia.com", "asianage.com", "newindianexpress.com", "orissapost.com",
    "punemirror.indiatimes.com", "mumbaimirror.indiatimes.com", "bangaloremirror.indiatimes.com", "ahmedabadmirror.com"
]

def get_all_top_urls_orchestrator(base_query_for_api: str, num_api_results_per_query_config: int, date_restrict_api_param: str | None, trusted_domains_for_query: list[str], location: str | None) -> list[dict]:
    """
    Orchestrates building refined queries and collecting unique URLs from Google.
    """
    is_entity_search = base_query_for_api.startswith('"') and base_query_for_api.endswith('"')
    # Pass location to build_google_search_queries if it needs to construct queries like "{entity} in {location} site:{domain}"
    refined_queries = build_google_search_queries(base_query_for_api, is_entity_search, trusted_domains_for_query) # Pass trusted_domains
    
    all_urls_info_dict = {} 
    urls_collected_count = 0

    for q_idx, q in enumerate(refined_queries):
        if urls_collected_count >= TOTAL_URLS_TO_PROCESS_LIMIT:
            logger.info(f"https://www.merriam-webster.com/dictionary/collector Reached total URL processing limit of {TOTAL_URLS_TO_PROCESS_LIMIT}.")
            break
        
        current_num_results_to_request = num_api_results_per_query_config
        if is_entity_search and q == base_query_for_api: 
            current_num_results_to_request = min(10, num_api_results_per_query_config + 2) 
        elif "site:" in q: 
            current_num_results_to_request = max(1, num_api_results_per_query_config // 2)

        urls_from_api_for_q = []
        for attempt in range(MAX_API_RETRIES): 
            urls_from_api_for_q = get_urls_from_google_api(q, current_num_results_to_request, date_restrict_api_param)
            if urls_from_api_for_q: 
                break 
            logger.warning(f"  Attempt {attempt+1} for query '{q}' returned no results. Retrying if possible...")
            if attempt < MAX_API_RETRIES - 1:
                time.sleep(BACKOFF_FACTOR_API * (2 ** attempt))
            else:
                logger.error(f"  Failed to get URLs for query '{q}' after {MAX_API_RETRIES} attempts.")
        
        for url_info in urls_from_api_for_q:
            if url_info['url'] not in all_urls_info_dict: 
                all_urls_info_dict[url_info['url']] = url_info 
                urls_collected_count +=1
                if urls_collected_count >= TOTAL_URLS_TO_PROCESS_LIMIT:
                    break
        
        if q_idx < len(refined_queries) - 1: 
            time.sleep(random.uniform(1.0, 1.5)) 

    unique_urls_list = list(all_urls_info_dict.values())
    
    # Domain Prioritization (from User's Step 1)
    is_general_news_query = "news" in base_query_for_api.lower() or "latest" in base_query_for_api.lower() or date_restrict_api_param is not None
    if is_general_news_query:
        logger.info("https://www.merriam-webster.com/dictionary/collector Applying domain prioritization for news query...")
        preferred_domains_map = { # Example, can be expanded
            'timesofindia.indiatimes.com': 10, 'hindustantimes.com': 9, 'indianexpress.com': 9,
            'ndtv.com': 8, 'news18.com': 8, 'thehindu.com': 8, 
            'livehindustan.com': 9, 'amarujala.com': 9, 'jagran.com': 9, 
            'bbc.com': 7, 'reuters.com': 7, 'apnews.com': 7,
            'espncricinfo.com': 10, 'cricbuzz.com': 10 
        }
        # Add location specific trusted domains to preferred map with high priority
        if location and "lucknow" in location.lower():
            for domain in TRUSTED_DOMAINS_LUCKNOW:
                if domain not in preferred_domains_map: preferred_domains_map[domain] = 10
        
        for url_info in unique_urls_list:
            domain = url_info.get('domain', '')
            priority = preferred_domains_map.get(domain, 0)
            if priority == 0: 
                for preferred_domain_key in preferred_domains_map:
                    if domain.endswith(preferred_domain_key):
                        priority = preferred_domains_map[preferred_domain_key] -1 
                        break
            url_info['priority'] = priority
        unique_urls_list.sort(key=lambda x: x.get('priority', 0), reverse=True)
        logger.info(f"  Sorted by domain priority. Top domains after sort: {[u.get('domain') for u in unique_urls_list[:5]]}")

    logger.info(f"https://www.merriam-webster.com/dictionary/collector Collected and prioritized {len(unique_urls_list)} unique URLs.")
    return unique_urls_list[:TOTAL_URLS_TO_PROCESS_LIMIT] 


def extract_article_content(url_info: dict) -> dict:
    # This function logic is kept the same as the robust version from google_search_scraper_py_no_trusted_domains_fixed_args
    # (Ensuring it handles datetime objects correctly for publish_date)
    url = url_info.get('url', 'N/A')
    title_from_search = url_info.get('title', 'N/A')
    domain = url_info.get('domain', 'N/A').lower() # Use lowercased domain for dict lookup
    
    article_data = {
        'url': url, 'title': title_from_search, 'text': '', 
        'publish_date': None, 'domain': domain, 
        'extraction_method': 'none', 
        'extraction_note': 'Processing not attempted or failed early.'
    }

    html_content = get_html_with_headers(url) 

    if not html_content:
        article_data['extraction_note'] = "Failed to download HTML content after all attempts."
        logger.error(f"  [Extractor] Failed to download HTML for {url}")
        return article_data

    # Try domain-specific custom extraction first (Step 3 & 4)
    if domain in CUSTOM_EXTRACTORS:
        try:
            custom_result = CUSTOM_EXTRACTORS[domain](html_content, url)
            if custom_result and custom_result.get('text') and len(custom_result.get('text')) >= MIN_CONTENT_LENGTH:
                logger.info(f"  [Custom Extractor] Success for {url} ({domain})")
                article_data.update({
                    'title': custom_result.get('title', article_data['title']),
                    'text': custom_result.get('text'),
                    'publish_date': custom_result.get('publish_date', article_data['publish_date']),
                    'extraction_method': custom_result.get('extraction_method', 'custom'),
                    'extraction_note': custom_result.get('extraction_note', f"Success (custom_{domain.split('.')[0]})")
                })
                # Ensure publish_date is datetime after custom extraction
                if article_data['publish_date'] and isinstance(article_data['publish_date'], str):
                    try:
                        article_data['publish_date'] = datetime.fromisoformat(article_data['publish_date'].replace('Z', '+00:00')).replace(tzinfo=timezone.utc)
                    except: pass # Keep as string if parsing fails
                return article_data
            elif custom_result and custom_result.get('text'):
                 logger.warning(f"  [Custom Extractor] Short content from {domain} for {url}. Length: {len(custom_result.get('text'))}")
                 article_data.update(custom_result) 
            else:
                logger.warning(f"  [Custom Extractor] Failed or no content from {domain} for {url}. Note: {custom_result.get('extraction_note', 'N/A')}")
        except Exception as e:
            logger.error(f"  [Custom Extractor] Error for {domain} on {url}: {str(e)}")
            article_data['extraction_note'] = f"Custom extractor error for {domain}"


    if not article_data.get('text') or len(article_data.get('text', '')) < MIN_CONTENT_LENGTH :
        try: 
            article = Article(url, config=NP_CONFIG) 
            article.html = html_content 
            article.parse()
            text = article.text.strip()
            
            current_title = article_data.get('title', title_from_search)
            if article.title and article.title.strip() and \
               (current_title == 'N/A' or not str(current_title).strip() or len(article.title.strip()) > len(str(current_title))):
                article_data['title'] = article.title.strip()
            
            if article.publish_date and isinstance(article.publish_date, datetime.datetime): 
                 article_data['publish_date'] = article.publish_date.replace(tzinfo=timezone.utc) if article.publish_date.tzinfo is None else article.publish_date.astimezone(timezone.utc)
            elif article.publish_date: 
                try: 
                    parsed_date_str = str(article.publish_date)
                    if 'T' in parsed_date_str: 
                        parsed_date = datetime.datetime.fromisoformat(parsed_date_str.replace('Z', '+00:00'))
                    else: 
                        parsed_date = datetime.datetime.strptime(parsed_date_str, '%Y-%m-%d %H:%M:%S')
                    article_data['publish_date'] = parsed_date.replace(tzinfo=timezone.utc) if parsed_date.tzinfo is None else parsed_date.astimezone(timezone.utc)
                except (ValueError, TypeError):
                    logger.warning(f"  [newspaper3k] Could not parse publish_date: {article.publish_date} (type: {type(article.publish_date)}) for {url}")
                    article_data['publish_date'] = article_data.get('publish_date', None) # Keep from custom if exists

            if text and len(text) >= MIN_CONTENT_LENGTH:
                article_data['text'] = text; article_data['extraction_method'] = 'newspaper3k'
                article_data['extraction_note'] = f"Success (newspaper3k) ~{len(text)} chars."
                logger.info(f"  [newspaper3k] Success for {url} (Length: {len(text)})")
                return article_data
            elif text: 
                if not article_data.get('text') or len(text) > len(article_data.get('text','')):
                    article_data['text'] = text
                article_data['extraction_method'] = 'newspaper3k_short'
                article_data['extraction_note'] = f"{article_data.get('extraction_note','')} | newspaper3k short text ({len(text)} chars)."
            else: 
                article_data['extraction_note'] = f"{article_data.get('extraction_note','')} | newspaper3k extracted no text."
                
        except ArticleException as e:
            logger.error(f"  [newspaper3k] ArticleException for {url}: {e}")
            article_data['extraction_note'] = f"{article_data.get('extraction_note','')} | newspaper3k ArticleException: {str(e)[:100]}"
        except Exception as e:
            logger.error(f"  [newspaper3k] General error for {url}: {e}")
            article_data['extraction_note'] = f"{article_data.get('extraction_note','')} | newspaper3k general error: {str(e)[:100]}"

    if not article_data.get('text') or len(article_data.get('text', '')) < MIN_CONTENT_LENGTH:
        original_extraction_note = article_data.get('extraction_note', 'Previous methods failed.')
        logger.info(f"  [Fallback BS4] For: {url} (Previous note: {original_extraction_note})")
        try:
            soup = BeautifulSoup(html_content, 'lxml') 
            for element in soup(['script', 'style', 'nav', 'footer', 'aside', 'header', '.sidebar', '#sidebar', 'form', 'button', 'input', '.ad', '.advertisement', '.cookie-banner', 'img', 'figure', 'figcaption', 'iframe', 'link', 'meta']):
                element.decompose()
            main_content_text = ""
            main_selectors = ['article[class*="content"]', 'div[class*="content"]', 'article[id*="content"]', 'div[id*="content"]', 'article', 'main', '.main-content', '#main', '.post-body', '.entry-content', '.td-post-content', '.story-content', '[role="main"]', '.article-body', '.articleBody']
            for selector in main_selectors:
                content_area = soup.select_one(selector)
                if content_area:
                    main_content_text = content_area.get_text(separator=' ', strip=True)
                    if len(main_content_text) >= MIN_CONTENT_LENGTH: 
                        logger.info(f"    [BeautifulSoup] Found good content with selector: {selector}")
                        break 
            if len(main_content_text) < MIN_CONTENT_LENGTH: 
                body = soup.find('body')
                body_text = body.get_text(separator=' ', strip=True) if body else ""
                final_bs_text = main_content_text if main_content_text and len(main_content_text) > len(body_text) / 3 else body_text
            else: 
                final_bs_text = main_content_text
            final_bs_text = ' '.join(final_bs_text.split()) 
            
            if final_bs_text and len(final_bs_text) >= MIN_CONTENT_LENGTH:
                article_data['text'] = final_bs_text
                article_data['extraction_method'] = 'beautifulsoup'
                article_data['extraction_note'] = f"Success (BS4) ~{len(final_bs_text)} chars."
                logger.info(f"    [BeautifulSoup] Success for {url} (Length: {len(final_bs_text)})")
            elif final_bs_text: 
                if not article_data.get('text') or len(final_bs_text) > len(article_data.get('text','')):
                    article_data['text'] = final_bs_text
                article_data['extraction_method'] = 'beautifulsoup_short'
                article_data['extraction_note'] = f"{original_extraction_note} | BS4 short text ({len(final_bs_text)} chars)." 
                logger.warning(f"    [BeautifulSoup] Short content for {url} (Length: {len(final_bs_text)})")
            else: 
                article_data['extraction_note'] = f"{original_extraction_note} | BS4 no significant text." 
                logger.warning(f"    [BeautifulSoup] No significant text for {url}.")
        except Exception as e:
            logger.error(f"  [Fallback BS4] Error for {url}: {e}")
            article_data['extraction_note'] = f"{article_data.get('extraction_note','')} | BS4 error: {str(e)[:100]}"
            if not article_data.get('text') or str(article_data.get('text','')).isspace():
                 article_data['text'] = ''
                 
    if (article_data.get('title') == 'N/A' or not str(article_data.get('title','')).strip()) and article_data.get('text','').strip(): 
        article_data['title'] = ' '.join(article_data.get('text','').split()[:12]) + "..."

    # Ensure publish_date is a datetime object for consistent filtering later
    if article_data['publish_date'] and isinstance(article_data['publish_date'], str):
        try:
            article_data['publish_date'] = datetime.fromisoformat(article_data['publish_date'].replace('Z', '+00:00')).replace(tzinfo=timezone.utc)
        except: # If parsing as ISO string fails, set to None
            logger.warning(f"Final attempt to parse publish_date string '{article_data['publish_date']}' failed for {url}")
            article_data['publish_date'] = None
    elif article_data['publish_date'] and isinstance(article_data['publish_date'], datetime.datetime):
        # Ensure it's timezone aware (UTC)
        if article_data['publish_date'].tzinfo is None:
            article_data['publish_date'] = article_data['publish_date'].replace(tzinfo=timezone.utc)
        else:
            article_data['publish_date'] = article_data['publish_date'].astimezone(timezone.utc)


    return article_data


def detect_trending_topics(articles: list[dict]) -> dict:
    """Identify trending topics from article titles to prioritize important stories."""
    all_text_for_trending = ""
    for article in articles: 
        title = article.get('title', '')
        all_text_for_trending += f" {title}" 

    if not all_text_for_trending.strip():
        return {}
    words = re.findall(r'\b[A-Za-z]{4,}\b', all_text_for_trending.lower()) 
    stopwords = { # Expanded stopwords
        'the', 'and', 'for', 'that', 'this', 'with', 'from', 'what', 'have', 'news', 'latest', 'updates',
        'google', 'search', 'results', 'article', 'articles', 'content', 'more', 'about', 'also', 'been',
        'could', 'after', 'into', 'their', 'them', 'then', 'there', 'these', 'they', 'were', 'will',
        'would', 'year', 'years', 'says', 'said', 'like', 'just', 'city', 'today', 'time', 'times', 'india',
        'indian', 'lucknow', 'delhi', 'mumbai', 'sport', 'sports', 'cricket', 'ipl', 'team', 'teams',
        'live', 'highlights', 'match', 'points', 'table', 'current', 'situation', 'veda', 'learning', 'center',
        'summer', 'carnival', 'check', 'schedule', 'player', 'result', 'ranking', 'admission', 'fees',
        'review', 'official', 'information', 'guide', 'explained', 'beyond', 'revolutionizing', 'rise',
        'state', 'beyond', 'future', 'trends', 'deep', 'dive', 'transformation', 'solutions', 'blog',
        'report', 'conference', 'event', 'events', 'technologies', 'technology', 'services', 'service'
    }
    word_counts = Counter(words)
    trending = {word: count for word, count in word_counts.items() if word not in stopwords and count > 1} 
    sorted_trending = dict(sorted(trending.items(), key=lambda item: item[1], reverse=True)[:5])
    logger.info(f"Detected trending topics: {sorted_trending}")
    return sorted_trending


def get_content_from_google_search(base_llm_query: str, location: str = None, lookback_hours: int = None) -> list[dict]:
    """
    Main orchestrator for Google Search.
    """
    logger.info(f"--- Starting Google Web Search & Extraction for base query: '{base_llm_query}' ---")
    
    query_for_api, relevant_trusted_domains, num_api_res_per_q, date_restrict_api = \
        determine_query_params_for_google(base_llm_query, location, lookback_hours)

    urls_info_list = get_all_top_urls_orchestrator(
        query_for_api, # This is the query potentially modified with location and quotes
        num_api_res_per_q,
        date_restrict_api,
        relevant_trusted_domains, # Pass the context-specific trusted domains
        location # Pass location for query building inside orchestrator
    )
    
    if not urls_info_list:
        logger.warning(f"[Pipeline] No URLs found by Google Search for base query: '{base_llm_query}' after attempting refined queries.")
        # Diagnostic: Provide more details for debugging
        logger.warning(f"[Pipeline] Diagnostics: query_for_api='{query_for_api}', relevant_trusted_domains={relevant_trusted_domains}, num_api_res_per_q={num_api_res_per_q}, date_restrict_api={date_restrict_api}")
        return []
            
    logger.info(f"[Pipeline] Processing {len(urls_info_list)} unique URLs for content extraction.")
    
    processed_articles_before_filter = []
    for url_info_item in urls_info_list:
        article_data = extract_article_content(url_info_item)
        # Diagnostic: Log extraction note and content length
        logger.info(f"[Pipeline] Extraction note: {article_data.get('extraction_note','N/A')} | Content length: {len(article_data.get('text',''))}")
        processed_articles_before_filter.append(article_data)

    # Trending topics detection
    if "news" in base_llm_query.lower() or "latest" in base_llm_query.lower() or lookback_hours is not None:
        trending_topics = detect_trending_topics(processed_articles_before_filter)
        if trending_topics:
            for article in processed_articles_before_filter:
                article['trending_score'] = 0
                title_text = (article.get('title', '') + " " + article.get('text', '')[:500]).lower() 
                for topic, count in trending_topics.items():
                    if topic in title_text:
                        article['trending_score'] += count
            processed_articles_before_filter.sort(key=lambda x: (x.get('trending_score', 0), x.get('publish_date') is not None), reverse=True)
            logger.info(f"  Articles re-sorted by trending topics. Top trending scores: {[a.get('trending_score',0) for a in processed_articles_before_filter[:3]]}")


    final_results_for_llm = []
    if lookback_hours is not None:
        logger.info(f"[Pipeline] Applying '{lookback_hours} hours' post-filtering (Google)...")
        now_utc = datetime.now(timezone.utc) 
        cutoff_time = now_utc - timedelta(hours=lookback_hours)
        
        articles_with_valid_recent_dates = []
        articles_without_valid_dates_or_older = []

        for article in processed_articles_before_filter:
            publish_date_dt = article.get('publish_date') 
            if publish_date_dt and isinstance(publish_date_dt, datetime.datetime): 
                if publish_date_dt.tzinfo is None or publish_date_dt.tzinfo.utcoffset(publish_date_dt) is None: 
                    publish_date_dt = publish_date_dt.replace(tzinfo=timezone.utc) 
                else: 
                    publish_date_dt = publish_date_dt.astimezone(timezone.utc)
                article['publish_date'] = publish_date_dt # Store the timezone-aware version

                if publish_date_dt >= cutoff_time:
                    articles_with_valid_recent_dates.append(article)
                else:
                    logger.info(f"  Filtered out by date (older than {lookback_hours}h): {article.get('url')} (Published: {publish_date_dt.strftime('%Y-%m-%d %H:%M:%S %Z')})")
            else: 
                articles_without_valid_dates_or_older.append(article)
        
        final_results_for_llm.extend(articles_with_valid_recent_dates)
        needed_more = TOTAL_URLS_TO_PROCESS_LIMIT - len(final_results_for_llm)

        if needed_more > 0:
            articles_without_valid_dates_or_older.sort(key=lambda x: (x.get('trending_score', 0), len(x.get('text',''))), reverse=True)
            
            for article in articles_without_valid_dates_or_older:
                if len(final_results_for_llm) >= TOTAL_URLS_TO_PROCESS_LIMIT: break
                if article.get('text') and len(article.get('text')) >= MIN_CONTENT_LENGTH:
                    text_lower = article.get('text', '').lower()[:1000] # Check first 1000 chars
                    today_dt = datetime.now(timezone.utc) # Use timezone aware for today
                    today_str_formats = [today_dt.strftime('%B %d').lower(), today_dt.strftime('%d %B').lower()]
                    yesterday_dt = today_dt - timedelta(days=1)
                    yesterday_str_formats = [yesterday_dt.strftime('%B %d').lower(), yesterday_dt.strftime('%d %B').lower()]
                    
                    recency_hints = ['today', 'breaking', 'latest', 'hours ago', 'just now', 
                                     'this morning', 'this afternoon', 'this evening', 'yesterday'] + \
                                    today_str_formats + yesterday_str_formats
                    
                    if any(hint in text_lower for hint in recency_hints):
                        logger.warning(f"  Article kept despite '{lookback_hours}h' filter due to missing date but good content & recency hints: {article.get('url')}")
                        final_results_for_llm.append(article)
                    elif article.get('trending_score',0) > 0:
                         logger.warning(f"  Article kept despite '{lookback_hours}h' filter due to missing date but good content & trending: {article.get('url')}")
                         final_results_for_llm.append(article)
                    else:
                        logger.info(f"  Filtered out (no valid date, no strong recency hints/trends for time-bound query): {article.get('url')}")
                else:
                     logger.info(f"  Filtered out (no valid date & insufficient content): {article.get('url')}")
        logger.info(f"[Pipeline] Retained {len(final_results_for_llm)} articles after '{lookback_hours} hours' filter (Google).")
    else: 
        final_results_for_llm = [
            article for article in processed_articles_before_filter 
            if article.get('text') and len(article.get('text')) >= MIN_CONTENT_LENGTH
        ]
        if any('trending_score' in article for article in final_results_for_llm):
            final_results_for_llm.sort(key=lambda x: x.get('trending_score', 0), reverse=True)
        logger.info(f"[Pipeline] Retained {len(final_results_for_llm)} articles with substantial content (no time filter, Google).")

    final_results_for_llm = final_results_for_llm[:TOTAL_URLS_TO_PROCESS_LIMIT]
    successful_extractions = len([a for a in final_results_for_llm if a.get('text') and len(a['text']) >= MIN_CONTENT_LENGTH])
    logger.info(f"[Pipeline] Final processing complete. Returning {successful_extractions} articles with substantial text (Google).")
    return final_results_for_llm


if __name__ == '__main__':
    logger.info("--- Testing Google Search Scraper Module (Comprehensive Enhancements v2) ---")
    
    if not GOOGLE_API_KEY or not CUSTOM_SEARCH_ENGINE_ID:
        logger.error("\nERROR: GOOGLE_API_KEY and CUSTOM_SEARCH_ENGINE_ID environment variables must be set to run tests.")
    else:
        test_cases = [
            {"query": "news in Lucknow last 24 hours", "location": "Lucknow", "hours": 24},
            {"query": "Summer Carnival Veda Learning Center what is it about", "location": "Lucknow", "hours": None},
            {"query": f"IPL {str(datetime.now().year)} latest news", "location": None, "hours": 7*24} 
        ]
        
        for tc in test_cases:
            logger.info(f"\n\n--- Processing Test Query: '{tc['query']}' (Location: {tc['location']}, Lookback: {tc['hours']}h) ---")
            articles = get_content_from_google_search(
                tc['query'], 
                location=tc['location'],
                lookback_hours=tc['hours']
            )
            if articles:
                logger.info(f"\n[Test Result] Successfully processed {len(articles)} articles for query '{tc['query']}'.")
                for i, article in enumerate(articles):
                    logger.info(f"\n--- Article {i+1} ---")
                    logger.info(f"  URL: {article.get('url')}")
                    logger.info(f"  Title: {article.get('title')}")
                    pub_date_display = article.get('publish_date')
                    if isinstance(pub_date_display, datetime.datetime): 
                        pub_date_display = pub_date_display.strftime('%Y-%m-%d %H:%M:%S %Z')
                    logger.info(f"  Published: {pub_date_display}")
                    logger.info(f"  Domain: {article.get('domain')}")
                    logger.info(f"  Extraction Method: {article.get('extraction_method')}")
                    logger.info(f"  Extraction Note: {article.get('extraction_note')}")
                    logger.info(f"  Text (first 150 chars): {article.get('text', '')[:150].replace(os.linesep, ' ')}...")
            else:
                logger.info(f"[Test Result] No articles retrieved for query: '{tc['query']}'.")
            logger.info("--- End of Test Query ---")
            time.sleep(3)
