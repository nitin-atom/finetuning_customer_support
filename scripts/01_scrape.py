#!/usr/bin/env python3
"""
Phase 1: Scrape all articles from Atom helpdesk.

Outputs: data/raw/articles.json
"""

import os
import sys
import json
import logging
import argparse
from pathlib import Path
from datetime import datetime

import yaml
from tqdm import tqdm

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.scraper import Scraper

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    handlers=[
        logging.FileHandler('scrape.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def load_config() -> dict:
    """Load configuration from config.yaml."""
    config_path = Path(__file__).parent.parent / 'config' / 'config.yaml'
    with open(config_path) as f:
        return yaml.safe_load(f)


def load_checkpoint(checkpoint_path: Path) -> dict:
    """Load checkpoint if it exists."""
    if checkpoint_path.exists():
        with open(checkpoint_path) as f:
            return json.load(f)
    return {}


def save_checkpoint(checkpoint_path: Path, data: dict):
    """Save checkpoint data."""
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    with open(checkpoint_path, 'w') as f:
        json.dump(data, f, indent=2)


def save_articles(output_path: Path, articles: list[dict]):
    """Save articles to JSON file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(articles, f, indent=2)
    logger.info(f"Saved {len(articles)} articles to {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Scrape Atom helpdesk articles')
    parser.add_argument('--limit', type=int, help='Limit number of articles to scrape (for testing)')
    parser.add_argument('--resume', action='store_true', help='Resume from checkpoint')
    args = parser.parse_args()
    
    config = load_config()
    base_path = Path(__file__).parent.parent
    output_path = base_path / config['paths']['raw_articles']
    checkpoint_path = base_path / config['paths']['checkpoint']
    
    scraper = Scraper(config)
    
    # Load checkpoint if resuming
    checkpoint = {}
    scraped_articles = []
    processed_article_ids = set()
    
    if args.resume:
        checkpoint = load_checkpoint(checkpoint_path)
        if checkpoint.get('phase') == 'scraping' and checkpoint.get('articles'):
            scraped_articles = checkpoint['articles']
            processed_article_ids = {a['article_id'] for a in scraped_articles}
            logger.info(f"Resuming from checkpoint: {len(scraped_articles)} articles already scraped")
    
    # Step 1: Fetch homepage and extract collections
    logger.info("Fetching homepage...")
    homepage_html = scraper.fetch_page(config['scraping']['base_url'])
    if not homepage_html:
        logger.error("Failed to fetch homepage")
        return 1
    
    collections = scraper.extract_collections(homepage_html)
    logger.info(f"Found {len(collections)} collections")
    
    # Step 2: Extract all article URLs from each collection
    all_articles = []
    for collection in tqdm(collections, desc="Fetching collections"):
        logger.info(f"Fetching collection: {collection['name']}")
        collection_html = scraper.fetch_page(collection['url'])
        if collection_html:
            articles = scraper.extract_articles_from_collection(collection_html, collection)
            all_articles.extend(articles)
    
    logger.info(f"Found {len(all_articles)} total articles across all collections")
    
    # Remove duplicates (same article can appear in multiple collections)
    seen_ids = set()
    unique_articles = []
    for article in all_articles:
        if article['id'] not in seen_ids:
            seen_ids.add(article['id'])
            unique_articles.append(article)
    
    logger.info(f"Found {len(unique_articles)} unique articles after deduplication")
    
    # Apply limit if specified
    articles_to_scrape = unique_articles
    if args.limit:
        articles_to_scrape = unique_articles[:args.limit]
        logger.info(f"Limiting to {args.limit} articles for testing")
    
    # Step 3: Scrape each article
    for article_info in tqdm(articles_to_scrape, desc="Scraping articles"):
        # Skip if already processed
        if article_info['id'] in processed_article_ids:
            continue
        
        logger.debug(f"Scraping article: {article_info['title']}")
        article_html = scraper.fetch_page(article_info['url'])
        
        if article_html:
            try:
                article_data = scraper.extract_article_content(article_html, article_info)
                scraped_articles.append(article_data)
                processed_article_ids.add(article_info['id'])
                
                # Save checkpoint every 10 articles
                if len(scraped_articles) % 10 == 0:
                    save_checkpoint(checkpoint_path, {
                        'phase': 'scraping',
                        'last_updated': datetime.now().isoformat(),
                        'articles_scraped': len(scraped_articles),
                        'articles_total': len(articles_to_scrape),
                        'articles': scraped_articles
                    })
            except Exception as e:
                logger.error(f"Error extracting article {article_info['id']}: {e}")
        else:
            logger.warning(f"Failed to fetch article: {article_info['url']}")
    
    # Save final output
    save_articles(output_path, scraped_articles)
    
    # Update checkpoint
    save_checkpoint(checkpoint_path, {
        'phase': 'scraping_complete',
        'last_updated': datetime.now().isoformat(),
        'articles_scraped': len(scraped_articles),
        'articles_total': len(articles_to_scrape)
    })
    
    # Print summary
    print(f"\n{'='*50}")
    print("SCRAPING COMPLETE")
    print(f"{'='*50}")
    print(f"Collections processed: {len(collections)}")
    print(f"Articles scraped: {len(scraped_articles)}")
    print(f"Output file: {output_path}")
    
    # Collection breakdown
    collection_counts = {}
    for article in scraped_articles:
        col = article['collection']
        collection_counts[col] = collection_counts.get(col, 0) + 1
    
    print(f"\nArticles by collection:")
    for col, count in sorted(collection_counts.items(), key=lambda x: -x[1]):
        print(f"  - {col}: {count}")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
