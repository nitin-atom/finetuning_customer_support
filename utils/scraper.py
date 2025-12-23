"""
Scraper utilities for extracting content from Atom helpdesk.
"""

import re
import time
import logging
import requests
from bs4 import BeautifulSoup
import html2text
from typing import Optional

logger = logging.getLogger(__name__)


class Scraper:
    """Scraper for Atom helpdesk articles."""
    
    def __init__(self, config: dict):
        self.config = config
        self.base_url = config['scraping']['base_url']
        self.delay = config['scraping']['request_delay_seconds']
        self.max_retries = config['scraping']['max_retries']
        self.timeout = config['scraping']['timeout_seconds']
        self.user_agent = config['scraping']['user_agent']
        
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': self.user_agent,
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
        })
        
        self.html_converter = html2text.HTML2Text()
        self.html_converter.ignore_links = False
        self.html_converter.ignore_images = True
        self.html_converter.body_width = 0  # No wrapping
        
        self._last_request_time = 0
    
    def _rate_limit(self):
        """Enforce rate limiting between requests."""
        elapsed = time.time() - self._last_request_time
        if elapsed < self.delay:
            time.sleep(self.delay - elapsed)
        self._last_request_time = time.time()
    
    def fetch_page(self, url: str) -> Optional[str]:
        """Fetch a page with retry logic."""
        for attempt in range(self.max_retries):
            try:
                self._rate_limit()
                response = self.session.get(url, timeout=self.timeout)
                response.raise_for_status()
                return response.text
            except requests.RequestException as e:
                logger.warning(f"Attempt {attempt + 1} failed for {url}: {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff
                else:
                    logger.error(f"Failed to fetch {url} after {self.max_retries} attempts")
                    return None
        return None
    
    # Known collection name mappings for clean names
    COLLECTION_NAMES = {
        'atom-com-registrar': 'Atom.com Registrar',
        'domain-marketplace-for-sellers': 'Domain Marketplace for Sellers', 
        'domain-marketplace-for-buyers': 'Domain Marketplace for Buyers',
        'white-label-marketplace-for-sellers': 'White Label Marketplace For Sellers',
        'atompay': 'AtomPay',
        'atom-partnerships-atomconnect': 'Atom Partnerships (AtomConnect)',
        'atom-ai-tools': 'Atom AI Tools',
        'starting-a-naming-contest': 'Starting a Naming Contest',
        'creatives': 'Creatives'
    }
    
    def extract_collections(self, homepage_html: str) -> list[dict]:
        """Extract collection URLs from the homepage."""
        soup = BeautifulSoup(homepage_html, 'lxml')
        collections = []
        
        # Find all collection links
        for link in soup.select('a[href*="/collections/"]'):
            href = link.get('href', '')
            if '/collections/' in href and href.startswith(('http', '/')):
                # Extract collection info
                url = href if href.startswith('http') else f"https://helpdesk.atom.com{href}"
                
                # Extract collection ID from URL
                match = re.search(r'/collections/(\d+)-(.+?)(?:\?|$|#)', url)
                if match:
                    collection_id = match.group(1)
                    slug = match.group(2)
                    
                    # Use known name mapping or generate from slug
                    name = self.COLLECTION_NAMES.get(slug, slug.replace('-', ' ').title())
                    
                    # Avoid duplicates
                    if not any(c['id'] == collection_id for c in collections):
                        collections.append({
                            'id': collection_id,
                            'slug': slug,
                            'name': name,
                            'url': url.split('?')[0]  # Remove query params
                        })
        
        logger.info(f"Found {len(collections)} collections")
        return collections
    
    def extract_articles_from_collection(self, collection_html: str, collection_info: dict) -> list[dict]:
        """Extract all article URLs from a collection page."""
        soup = BeautifulSoup(collection_html, 'lxml')
        articles = []
        
        # Find all article links
        for link in soup.select('a[href*="/articles/"]'):
            href = link.get('href', '')
            if '/articles/' in href:
                url = href if href.startswith('http') else f"https://helpdesk.atom.com{href}"
                
                # Extract article ID
                match = re.search(r'/articles/(\d+)-(.+?)(?:\?|$|#)', url)
                if match:
                    article_id = match.group(1)
                    slug = match.group(2)
                    
                    # Get title from link text
                    title = link.get_text(strip=True)
                    
                    # Avoid duplicates
                    if not any(a['id'] == article_id for a in articles):
                        articles.append({
                            'id': article_id,
                            'slug': slug,
                            'title': title,
                            'url': url.split('?')[0],
                            'collection': collection_info['name'],
                            'collection_id': collection_info['id']
                        })
        
        logger.info(f"Found {len(articles)} articles in collection '{collection_info['name']}'")
        return articles
    
    def extract_article_content(self, article_html: str, article_info: dict) -> dict:
        """Extract structured content from an article page."""
        soup = BeautifulSoup(article_html, 'lxml')
        
        # Find the main article content
        article_body = soup.select_one('article') or soup.select_one('.article-body') or soup.select_one('main')
        
        if not article_body:
            logger.warning(f"Could not find article body for {article_info['url']}")
            article_body = soup.body or soup
        
        # Remove navigation, footer, and other non-content elements
        for selector in ['nav', 'footer', 'header', '.intercom-reaction', '.intercom-article-meta', 
                         '[data-testid="article-footer"]', '.article-footer', 'script', 'style']:
            for element in article_body.select(selector):
                element.decompose()
        
        # Get the title
        title_elem = soup.select_one('h1') or soup.select_one('title')
        title = title_elem.get_text(strip=True) if title_elem else article_info.get('title', '')
        # Clean title
        title = re.sub(r'\s*\|\s*Atom Help Center.*$', '', title)
        
        # Get description from meta tag
        meta_desc = soup.select_one('meta[name="description"]') or soup.select_one('meta[property="og:description"]')
        description = meta_desc.get('content', '') if meta_desc else ''
        
        # Convert to markdown
        raw_html = str(article_body)
        markdown = self.html_converter.handle(raw_html)
        
        # Clean up markdown
        markdown = self._clean_markdown(markdown)
        
        # Extract plain text
        plain_text = article_body.get_text(separator=' ', strip=True)
        plain_text = re.sub(r'\s+', ' ', plain_text).strip()
        
        # Extract sections (headers and their content)
        sections = self._extract_sections(article_body)
        
        # Find related articles
        related = []
        for link in soup.select('a[href*="/articles/"]'):
            href = link.get('href', '')
            link_text = link.get_text(strip=True)
            if href and link_text and href != article_info['url']:
                url = href if href.startswith('http') else f"https://helpdesk.atom.com{href}"
                if not any(r['url'] == url for r in related):
                    related.append({'title': link_text, 'url': url.split('?')[0]})
        
        # Get metadata
        word_count = len(plain_text.split())
        has_images = bool(soup.select('img'))
        has_tables = bool(soup.select('table'))
        has_video = bool(soup.select('video, iframe[src*="youtube"], iframe[src*="vimeo"]'))
        
        return {
            'article_id': article_info['id'],
            'url': article_info['url'],
            'title': title,
            'description': description,
            'collection': article_info['collection'],
            'collection_id': article_info['collection_id'],
            'content': {
                'raw_html': raw_html,
                'markdown': markdown,
                'plain_text': plain_text,
                'sections': sections
            },
            'related_articles': related[:5],  # Limit to 5
            'metadata': {
                'word_count': word_count,
                'has_images': has_images,
                'has_tables': has_tables,
                'has_video': has_video
            }
        }
    
    def _clean_markdown(self, markdown: str) -> str:
        """Clean up converted markdown."""
        # Remove excessive newlines
        markdown = re.sub(r'\n{3,}', '\n\n', markdown)
        
        # Remove leftover HTML tags
        markdown = re.sub(r'<[^>]+>', '', markdown)
        
        # Clean up link formatting
        markdown = re.sub(r'\[([^\]]+)\]\(javascript:[^)]*\)', r'\1', markdown)
        
        # Remove "Skip to main content" and similar
        markdown = re.sub(r'Skip to main content\n*', '', markdown)
        
        # Trim whitespace
        markdown = markdown.strip()
        
        return markdown
    
    def _extract_sections(self, article_body) -> list[dict]:
        """Extract sections based on headers."""
        sections = []
        current_section = None
        current_content = []
        
        for element in article_body.descendants:
            if element.name in ['h1', 'h2', 'h3', 'h4']:
                # Save previous section
                if current_section:
                    sections.append({
                        'heading': current_section,
                        'level': current_section_level,
                        'content': ' '.join(current_content).strip()
                    })
                
                current_section = element.get_text(strip=True)
                current_section_level = int(element.name[1])
                current_content = []
            elif hasattr(element, 'name') and element.name in ['p', 'li', 'td', 'span'] and element.string:
                text = element.get_text(strip=True)
                if text and current_section:
                    current_content.append(text)
        
        # Save last section
        if current_section:
            sections.append({
                'heading': current_section,
                'level': current_section_level,
                'content': ' '.join(current_content).strip()
            })
        
        return sections
