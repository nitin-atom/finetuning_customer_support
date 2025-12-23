#!/usr/bin/env python3
"""
Phase 2: Generate questions for each article using OpenAI Batch API.

Inputs: data/raw/articles.json
Outputs: data/intermediate/questions.json
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

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.llm_client import LLMClient, load_prompt, format_prompt
from utils.validators import parse_questions_json

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    handlers=[
        logging.FileHandler('questions.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def load_config() -> dict:
    config_path = Path(__file__).parent.parent / 'config' / 'config.yaml'
    with open(config_path) as f:
        return yaml.safe_load(f)


def load_checkpoint(checkpoint_path: Path) -> dict:
    if checkpoint_path.exists():
        with open(checkpoint_path) as f:
            return json.load(f)
    return {}


def save_checkpoint(checkpoint_path: Path, data: dict):
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    with open(checkpoint_path, 'w') as f:
        json.dump(data, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description='Generate questions for articles')
    parser.add_argument('--limit', type=int, help='Limit number of articles (for testing)')
    parser.add_argument('--resume', action='store_true', help='Resume from checkpoint')
    parser.add_argument('--sync', action='store_true', help='Use synchronous API instead of batch')
    args = parser.parse_args()
    
    config = load_config()
    base_path = Path(__file__).parent.parent
    
    articles_path = base_path / config['paths']['raw_articles']
    output_path = base_path / config['paths']['questions']
    checkpoint_path = base_path / config['paths']['checkpoint']
    prompt_path = base_path / 'prompts' / 'question_generation.txt'
    
    # Load articles
    with open(articles_path) as f:
        articles = json.load(f)
    
    logger.info(f"Loaded {len(articles)} articles")
    
    # Apply limit
    if args.limit:
        articles = articles[:args.limit]
        logger.info(f"Limiting to {args.limit} articles")
    
    # Load checkpoint
    existing_questions = {}
    if args.resume:
        checkpoint = load_checkpoint(checkpoint_path)
        if checkpoint.get('phase') == 'questions' and checkpoint.get('questions'):
            existing_questions = checkpoint['questions']
            logger.info(f"Resuming: {len(existing_questions)} articles already processed")
    
    # Filter articles not yet processed
    articles_to_process = [a for a in articles if a['article_id'] not in existing_questions]
    logger.info(f"{len(articles_to_process)} articles to process")
    
    if not articles_to_process:
        logger.info("All articles already processed")
        return 0
    
    # Load prompt template
    prompt_template = load_prompt(prompt_path)
    
    # Initialize LLM client
    client = LLMClient(config)
    
    all_questions = dict(existing_questions)
    
    if args.sync:
        # Synchronous mode (for testing)
        for article in tqdm(articles_to_process, desc="Generating questions"):
            prompt = format_prompt(
                prompt_template,
                title=article['title'],
                collection=article['collection'],
                description=article.get('description', ''),
                content=article['content']['markdown'][:8000]  # Limit content size
            )
            
            response = client.generate_single(
                prompt,
                temperature=config['generation']['temperature_questions'],
                max_tokens=config['generation']['max_tokens_questions']
            )
            
            if response:
                questions = parse_questions_json(response)
                all_questions[article['article_id']] = {
                    'article_id': article['article_id'],
                    'title': article['title'],
                    'collection': article['collection'],
                    'questions': questions
                }
                
                # Save checkpoint
                if len(all_questions) % 5 == 0:
                    save_checkpoint(checkpoint_path, {
                        'phase': 'questions',
                        'last_updated': datetime.now().isoformat(),
                        'processed': len(all_questions),
                        'questions': all_questions
                    })
    else:
        # Batch mode
        logger.info("Preparing batch requests...")
        batch_requests = []
        
        for article in articles_to_process:
            prompt = format_prompt(
                prompt_template,
                title=article['title'],
                collection=article['collection'],
                description=article.get('description', ''),
                content=article['content']['markdown'][:8000]
            )
            
            batch_requests.append({
                'custom_id': article['article_id'],
                'prompt': prompt,
                'temperature': config['generation']['temperature_questions'],
                'max_tokens': config['generation']['max_tokens_questions']
            })
        
        # Run batch
        batch_file = base_path / 'data' / 'intermediate' / 'questions_batch.jsonl'
        logger.info(f"Submitting batch with {len(batch_requests)} requests...")
        
        results = client.run_batch(
            batch_requests,
            batch_file,
            description="Question generation for Atom helpdesk"
        )
        
        # Process results
        article_map = {a['article_id']: a for a in articles}
        
        for result in results:
            article_id = result['custom_id']
            article = article_map.get(article_id)
            
            if result['error']:
                logger.error(f"Error for article {article_id}: {result['error']}")
                continue
            
            questions = parse_questions_json(result['content'])
            
            if questions:
                all_questions[article_id] = {
                    'article_id': article_id,
                    'title': article['title'] if article else '',
                    'collection': article['collection'] if article else '',
                    'questions': questions
                }
            else:
                logger.warning(f"No valid questions for article {article_id}")
    
    # Save output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(all_questions, f, indent=2)
    
    # Update checkpoint
    save_checkpoint(checkpoint_path, {
        'phase': 'questions_complete',
        'last_updated': datetime.now().isoformat(),
        'articles_processed': len(all_questions)
    })
    
    # Print summary
    total_questions = sum(len(q['questions']) for q in all_questions.values())
    avg_questions = total_questions / len(all_questions) if all_questions else 0
    
    print(f"\n{'='*50}")
    print("QUESTION GENERATION COMPLETE")
    print(f"{'='*50}")
    print(f"Articles processed: {len(all_questions)}")
    print(f"Total questions: {total_questions}")
    print(f"Average questions per article: {avg_questions:.1f}")
    print(f"Output file: {output_path}")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
