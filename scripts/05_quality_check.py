#!/usr/bin/env python3
"""
Phase 5: Quality assurance for the training data.

Inputs: data/output/training_data.jsonl, data/intermediate/qa_pairs.json, data/raw/articles.json
Outputs: data/output/final_training_data.jsonl, data/output/quality_report.json
"""

import os
import sys
import json
import logging
import random
from pathlib import Path
from datetime import datetime

import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.validators import (
    validate_json_structure,
    validate_content_length,
    validate_answer_grounding,
    deduplicate_qa_pairs
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
)
logger = logging.getLogger(__name__)


def load_config() -> dict:
    config_path = Path(__file__).parent.parent / 'config' / 'config.yaml'
    with open(config_path) as f:
        return yaml.safe_load(f)


def main():
    config = load_config()
    base_path = Path(__file__).parent.parent
    
    training_path = base_path / config['paths']['training_data']
    qa_pairs_path = base_path / config['paths']['qa_pairs']
    articles_path = base_path / config['paths']['raw_articles']
    output_path = base_path / config['paths']['final_training_data']
    report_path = base_path / config['paths']['quality_report']
    metadata_path = base_path / config['paths']['metadata']
    
    # Load data
    with open(qa_pairs_path) as f:
        qa_pairs = json.load(f)
    
    with open(articles_path) as f:
        articles = json.load(f)
    article_map = {a['article_id']: a for a in articles}
    
    logger.info(f"Loaded {len(qa_pairs)} Q&A pairs for validation")
    
    # Track validation results
    results = {
        'total_examples': len(qa_pairs),
        'json_validity': {'passed': 0, 'failed': 0},
        'schema_compliance': {'passed': 0, 'failed': 0},
        'content_length': {'passed': 0, 'failed': 0},
        'grounding_check': {'passed': 0, 'failed': 0, 'issues': []}
    }
    
    # Validate JSONL format
    with open(training_path) as f:
        for i, line in enumerate(f):
            valid, error = validate_json_structure(line)
            if valid:
                results['json_validity']['passed'] += 1
                results['schema_compliance']['passed'] += 1
            else:
                results['json_validity']['failed'] += 1
                results['schema_compliance']['failed'] += 1
                logger.warning(f"Line {i}: {error}")
    
    # Validate content lengths
    for qa in qa_pairs:
        valid, error = validate_content_length(
            qa['question'],
            qa['answer'],
            config
        )
        if valid:
            results['content_length']['passed'] += 1
        else:
            results['content_length']['failed'] += 1
            logger.debug(f"Content length issue: {error}")
    
    # Check grounding on sample
    sample_rate = config['validation']['semantic_sample_rate']
    sample_size = max(1, int(len(qa_pairs) * sample_rate))
    sample = random.sample(qa_pairs, min(sample_size, len(qa_pairs)))
    
    logger.info(f"Checking grounding on {len(sample)} samples")
    
    for qa in sample:
        article = article_map.get(qa['article_id'])
        if article:
            article_content = article['content']['plain_text']
            is_grounded, issues = validate_answer_grounding(qa['answer'], article_content)
            
            if is_grounded:
                results['grounding_check']['passed'] += 1
            else:
                results['grounding_check']['failed'] += 1
                results['grounding_check']['issues'].append({
                    'qa_id': qa['qa_id'],
                    'issues': issues
                })
    
    # Deduplicate
    deduped, dedup_stats = deduplicate_qa_pairs(qa_pairs, config)
    
    # Filter out failed items
    failed_qa_ids = set()
    
    # Mark items that failed content length
    for qa in qa_pairs:
        valid, _ = validate_content_length(qa['question'], qa['answer'], config)
        if not valid:
            failed_qa_ids.add(qa['qa_id'])
    
    # Filter deduped list
    final_pairs = [qa for qa in deduped if qa['qa_id'] not in failed_qa_ids]
    
    logger.info(f"Final dataset: {len(final_pairs)} examples")
    
    # Write final output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        for qa in final_pairs:
            collection = qa.get('collection', '')
            prompt_key = config['collection_prompt_mapping'].get(collection, 'default')
            system_prompt = config['system_prompts'].get(prompt_key, config['system_prompts']['default'])
            
            formatted = {
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": qa['question']},
                    {"role": "assistant", "content": qa['answer']}
                ]
            }
            f.write(json.dumps(formatted) + '\n')
    
    # Calculate rates
    grounding_pass_rate = (
        results['grounding_check']['passed'] / 
        (results['grounding_check']['passed'] + results['grounding_check']['failed'])
        if (results['grounding_check']['passed'] + results['grounding_check']['failed']) > 0
        else 1.0
    )
    
    # Generate quality report
    report = {
        'validation_timestamp': datetime.now().isoformat(),
        'total_examples_generated': results['total_examples'],
        'examples_after_validation': len(final_pairs),
        'removal_reasons': {
            'duplicate_exact': dedup_stats['exact_duplicates_removed'],
            'duplicate_near': dedup_stats['near_duplicates_removed'],
            'content_length_invalid': results['content_length']['failed']
        },
        'automated_checks': {
            'json_validity': results['json_validity'],
            'schema_compliance': results['schema_compliance'],
            'content_length': results['content_length']
        },
        'semantic_checks_sample': {
            'sample_size': len(sample),
            'grounding_pass_rate': round(grounding_pass_rate, 3),
            'grounding_issues': results['grounding_check']['issues'][:10]  # First 10
        },
        'recommendations': []
    }
    
    # Add recommendations
    if grounding_pass_rate < 0.95:
        report['recommendations'].append(
            f"Grounding pass rate is {grounding_pass_rate:.1%}. Review flagged examples."
        )
    
    if dedup_stats['exact_duplicates_removed'] + dedup_stats['near_duplicates_removed'] > 10:
        report['recommendations'].append(
            "High duplicate count. Consider reviewing question generation prompts."
        )
    
    if not report['recommendations']:
        report['recommendations'].append("All checks passed. Dataset is ready for fine-tuning.")
    
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    # Update metadata
    if metadata_path.exists():
        with open(metadata_path) as f:
            metadata = json.load(f)
        metadata['validation_passed'] = grounding_pass_rate >= 0.95
        metadata['final_examples'] = len(final_pairs)
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
    
    # Print summary
    print(f"\n{'='*50}")
    print("QUALITY ASSURANCE COMPLETE")
    print(f"{'='*50}")
    print(f"Original examples: {results['total_examples']}")
    print(f"After validation: {len(final_pairs)}")
    print(f"Removed: {results['total_examples'] - len(final_pairs)}")
    
    print(f"\nValidation Results:")
    print(f"  - JSON validity: {results['json_validity']['passed']}/{results['total_examples']}")
    print(f"  - Content length: {results['content_length']['passed']}/{results['total_examples']}")
    print(f"  - Grounding check: {results['grounding_check']['passed']}/{len(sample)} (sample)")
    print(f"  - Grounding rate: {grounding_pass_rate:.1%}")
    
    print(f"\nDeduplication:")
    print(f"  - Exact duplicates removed: {dedup_stats['exact_duplicates_removed']}")
    print(f"  - Near duplicates removed: {dedup_stats['near_duplicates_removed']}")
    
    print(f"\nOutput files:")
    print(f"  - {output_path}")
    print(f"  - {report_path}")
    
    if report['recommendations']:
        print(f"\nRecommendations:")
        for rec in report['recommendations']:
            print(f"  - {rec}")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
