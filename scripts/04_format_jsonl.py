#!/usr/bin/env python3
"""
Phase 4: Format Q&A pairs as OpenAI fine-tuning JSONL.

Inputs: data/intermediate/qa_pairs.json
Outputs: data/output/training_data.jsonl, data/output/metadata.json
"""

import os
import sys
import json
import logging
from pathlib import Path
from datetime import datetime

import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
)
logger = logging.getLogger(__name__)


def load_config() -> dict:
    config_path = Path(__file__).parent.parent / 'config' / 'config.yaml'
    with open(config_path) as f:
        return yaml.safe_load(f)


def get_system_prompt(collection: str, config: dict) -> str:
    """Get the appropriate system prompt for a collection."""
    prompt_key = config['collection_prompt_mapping'].get(collection, 'default')
    return config['system_prompts'].get(prompt_key, config['system_prompts']['default'])


def format_as_messages(qa_pair: dict, system_prompt: str) -> dict:
    """Format a Q&A pair as OpenAI messages structure."""
    return {
        "messages": [
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": qa_pair['question']
            },
            {
                "role": "assistant",
                "content": qa_pair['answer']
            }
        ]
    }


def main():
    config = load_config()
    base_path = Path(__file__).parent.parent
    
    qa_pairs_path = base_path / config['paths']['qa_pairs']
    output_path = base_path / config['paths']['training_data']
    metadata_path = base_path / config['paths']['metadata']
    
    # Load Q&A pairs
    with open(qa_pairs_path) as f:
        qa_pairs = json.load(f)
    
    logger.info(f"Loaded {len(qa_pairs)} Q&A pairs")
    
    # Format as JSONL
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    collection_counts = {}
    question_type_counts = {}
    total_answer_chars = 0
    
    with open(output_path, 'w') as f:
        for qa_pair in qa_pairs:
            collection = qa_pair.get('collection', '')
            system_prompt = get_system_prompt(collection, config)
            
            formatted = format_as_messages(qa_pair, system_prompt)
            f.write(json.dumps(formatted) + '\n')
            
            # Track stats
            collection_counts[collection] = collection_counts.get(collection, 0) + 1
            qt = qa_pair.get('question_type', 'unknown')
            question_type_counts[qt] = question_type_counts.get(qt, 0) + 1
            total_answer_chars += len(qa_pair.get('answer', ''))
    
    logger.info(f"Wrote {len(qa_pairs)} examples to {output_path}")
    
    # Calculate stats
    avg_answer_length = total_answer_chars / len(qa_pairs) if qa_pairs else 0
    
    # Get unique articles
    unique_articles = len(set(qa['article_id'] for qa in qa_pairs))
    
    # Create metadata
    metadata = {
        "generated_at": datetime.now().isoformat(),
        "total_examples": len(qa_pairs),
        "source_articles": unique_articles,
        "avg_questions_per_article": len(qa_pairs) / unique_articles if unique_articles else 0,
        "collections_covered": [
            {"name": name, "examples": count}
            for name, count in sorted(collection_counts.items(), key=lambda x: -x[1])
        ],
        "question_type_distribution": question_type_counts,
        "avg_answer_length_chars": round(avg_answer_length, 1),
        "output_file": str(output_path),
        "validation_passed": None  # Will be set by quality check
    }
    
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    logger.info(f"Wrote metadata to {metadata_path}")
    
    # Print summary
    print(f"\n{'='*50}")
    print("FORMATTING COMPLETE")
    print(f"{'='*50}")
    print(f"Total examples: {len(qa_pairs)}")
    print(f"Source articles: {unique_articles}")
    print(f"Avg questions per article: {metadata['avg_questions_per_article']:.1f}")
    print(f"Avg answer length: {avg_answer_length:.0f} chars")
    print(f"\nOutput files:")
    print(f"  - {output_path}")
    print(f"  - {metadata_path}")
    
    print(f"\nExamples by collection:")
    for item in metadata['collections_covered'][:5]:
        print(f"  - {item['name']}: {item['examples']}")
    
    print(f"\nQuestion types:")
    for qt, count in sorted(question_type_counts.items(), key=lambda x: -x[1]):
        print(f"  - {qt}: {count}")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
