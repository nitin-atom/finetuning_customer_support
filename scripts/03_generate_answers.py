#!/usr/bin/env python3
"""
Phase 3: Generate answers for each question using OpenAI Batch API.

Inputs: data/raw/articles.json, data/intermediate/questions.json
Outputs: data/intermediate/qa_pairs.json
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

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    handlers=[
        logging.FileHandler('answers.log'),
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
    parser = argparse.ArgumentParser(description='Generate answers for questions')
    parser.add_argument('--limit', type=int, help='Limit number of Q&A pairs (for testing)')
    parser.add_argument('--resume', action='store_true', help='Resume from checkpoint')
    parser.add_argument('--sync', action='store_true', help='Use synchronous API instead of batch')
    args = parser.parse_args()
    
    config = load_config()
    base_path = Path(__file__).parent.parent
    
    articles_path = base_path / config['paths']['raw_articles']
    questions_path = base_path / config['paths']['questions']
    output_path = base_path / config['paths']['qa_pairs']
    checkpoint_path = base_path / config['paths']['checkpoint']
    prompt_path = base_path / 'prompts' / 'answer_generation.txt'
    
    # Load articles
    with open(articles_path) as f:
        articles = json.load(f)
    article_map = {a['article_id']: a for a in articles}
    
    # Load questions
    with open(questions_path) as f:
        questions_data = json.load(f)
    
    logger.info(f"Loaded {len(questions_data)} articles with questions")
    
    # Build list of all question-article pairs
    qa_items = []
    for article_id, data in questions_data.items():
        article = article_map.get(article_id)
        if not article:
            logger.warning(f"Article {article_id} not found in articles.json")
            continue
        
        for i, q in enumerate(data['questions']):
            qa_items.append({
                'qa_id': f"{article_id}_q{i}",
                'article_id': article_id,
                'question': q['question'],
                'question_type': q.get('type', 'unknown'),
                'article': article
            })
    
    logger.info(f"Total Q&A pairs to generate: {len(qa_items)}")
    
    # Apply limit
    if args.limit:
        qa_items = qa_items[:args.limit]
        logger.info(f"Limiting to {args.limit} Q&A pairs")
    
    # Load checkpoint
    existing_answers = {}
    if args.resume:
        checkpoint = load_checkpoint(checkpoint_path)
        if checkpoint.get('phase') == 'answers' and checkpoint.get('qa_pairs'):
            existing_answers = {qa['qa_id']: qa for qa in checkpoint['qa_pairs']}
            logger.info(f"Resuming: {len(existing_answers)} Q&A pairs already processed")
    
    # Filter items not yet processed
    items_to_process = [item for item in qa_items if item['qa_id'] not in existing_answers]
    logger.info(f"{len(items_to_process)} Q&A pairs to process")
    
    if not items_to_process:
        logger.info("All Q&A pairs already processed")
        return 0
    
    # Load prompt template
    prompt_template = load_prompt(prompt_path)
    
    # Initialize LLM client
    client = LLMClient(config)
    
    all_qa_pairs = list(existing_answers.values())
    
    if args.sync:
        # Synchronous mode (for testing)
        for item in tqdm(items_to_process, desc="Generating answers"):
            article = item['article']
            prompt = format_prompt(
                prompt_template,
                title=article['title'],
                collection=article['collection'],
                content=article['content']['markdown'][:8000],
                question=item['question']
            )
            
            response = client.generate_single(
                prompt,
                temperature=config['generation']['temperature_answers'],
                max_tokens=config['generation']['max_tokens_answers']
            )
            
            if response:
                qa_pair = {
                    'qa_id': item['qa_id'],
                    'article_id': item['article_id'],
                    'question': item['question'],
                    'question_type': item['question_type'],
                    'answer': response.strip(),
                    'collection': article['collection'],
                    'article_title': article['title']
                }
                all_qa_pairs.append(qa_pair)
                
                # Save checkpoint
                if len(all_qa_pairs) % 10 == 0:
                    save_checkpoint(checkpoint_path, {
                        'phase': 'answers',
                        'last_updated': datetime.now().isoformat(),
                        'processed': len(all_qa_pairs),
                        'qa_pairs': all_qa_pairs
                    })
    else:
        # Batch mode
        logger.info("Preparing batch requests...")
        batch_requests = []
        item_map = {}
        
        for item in items_to_process:
            article = item['article']
            prompt = format_prompt(
                prompt_template,
                title=article['title'],
                collection=article['collection'],
                content=article['content']['markdown'][:8000],
                question=item['question']
            )
            
            batch_requests.append({
                'custom_id': item['qa_id'],
                'prompt': prompt,
                'temperature': config['generation']['temperature_answers'],
                'max_tokens': config['generation']['max_tokens_answers']
            })
            item_map[item['qa_id']] = item
        
        # Run batch
        batch_file = base_path / 'data' / 'intermediate' / 'answers_batch.jsonl'
        logger.info(f"Submitting batch with {len(batch_requests)} requests...")
        
        results = client.run_batch(
            batch_requests,
            batch_file,
            description="Answer generation for Atom helpdesk"
        )
        
        # Process results
        for result in results:
            qa_id = result['custom_id']
            item = item_map.get(qa_id)
            
            if result['error']:
                logger.error(f"Error for {qa_id}: {result['error']}")
                continue
            
            if item:
                qa_pair = {
                    'qa_id': qa_id,
                    'article_id': item['article_id'],
                    'question': item['question'],
                    'question_type': item['question_type'],
                    'answer': result['content'].strip(),
                    'collection': item['article']['collection'],
                    'article_title': item['article']['title']
                }
                all_qa_pairs.append(qa_pair)
    
    # Save output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(all_qa_pairs, f, indent=2)
    
    # Update checkpoint
    save_checkpoint(checkpoint_path, {
        'phase': 'answers_complete',
        'last_updated': datetime.now().isoformat(),
        'qa_pairs_generated': len(all_qa_pairs)
    })
    
    # Print summary
    print(f"\n{'='*50}")
    print("ANSWER GENERATION COMPLETE")
    print(f"{'='*50}")
    print(f"Q&A pairs generated: {len(all_qa_pairs)}")
    print(f"Output file: {output_path}")
    
    # Question type breakdown
    type_counts = {}
    for qa in all_qa_pairs:
        qt = qa.get('question_type', 'unknown')
        type_counts[qt] = type_counts.get(qt, 0) + 1
    
    print(f"\nQuestion types:")
    for qt, count in sorted(type_counts.items(), key=lambda x: -x[1]):
        print(f"  - {qt}: {count}")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
