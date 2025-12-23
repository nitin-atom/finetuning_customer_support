#!/usr/bin/env python3
"""
Phase 6: Fine-tune the OpenAI model.

Inputs: data/output/final_training_data.jsonl
Outputs: data/output/finetune_job.json, data/output/finetuned_model.json
"""

import os
import sys
import json
import time
import logging
import argparse
from pathlib import Path

import yaml
from dotenv import load_dotenv
from openai import OpenAI

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


def setup_client():
    env_path = Path(__file__).parent.parent / '.env'
    load_dotenv(dotenv_path=env_path)
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        logger.error("OPENAI_API_KEY not found in .env file")
        sys.exit(1)
    return OpenAI(api_key=api_key)


def wait_for_file_processing(client, file_id):
    logger.info(f"Waiting for file {file_id} to be processed...")
    while True:
        file_obj = client.files.retrieve(file_id)
        if file_obj.status == 'processed':
            logger.info("File processed successfully.")
            return True
        elif file_obj.status == 'error':
            logger.error(f"File processing failed: {file_obj.status_details}")
            return False
        time.sleep(5)


def wait_for_job_completion(client, job_id, poll_interval=30):
    logger.info(f"Monitoring fine-tuning job {job_id}...")
    
    start_time = time.time()
    last_status = None
    
    while True:
        job = client.fine_tuning.jobs.retrieve(job_id)
        status = job.status
        
        if status != last_status:
            logger.info(f"Job status: {status}")
            last_status = status
            
        if status in ['succeeded', 'failed', 'cancelled']:
            return job
            
        # Log metrics if available (could be enhanced to show loss/accuracy)
        
        time.sleep(poll_interval)


def main():
    parser = argparse.ArgumentParser(description='Fine-tune OpenAI model')
    parser.add_argument('--dry-run', action='store_true', help='Validate files but do not start fine-tuning')
    args = parser.parse_args()
    
    config = load_config()
    client = setup_client()
    base_path = Path(__file__).parent.parent
    
    training_file_path = base_path / config['paths']['final_training_data']
    job_output_path = base_path / config['paths']['finetune_job']
    model_output_path = base_path / config['paths']['finetuned_model']
    
    # 1. Validate Training File
    if not training_file_path.exists():
        logger.error(f"Training file not found: {training_file_path}")
        return 1
        
    logger.info(f"Found training data: {training_file_path}")
    
    # 2. Upload File
    logger.info("Uploading training file...")
    if args.dry_run:
        logger.info("[DRY RUN] Would upload file here")
        file_id = "file-dummy-id"
    else:
        try:
            with open(training_file_path, "rb") as f:
                response = client.files.create(
                    file=f,
                    purpose="fine-tune"
                )
            file_id = response.id
            logger.info(f"File uploaded: {file_id}")
            
            if not wait_for_file_processing(client, file_id):
                return 1
                
        except Exception as e:
            logger.error(f"Failed to upload file: {e}")
            return 1

    # 3. Create Fine-tuning Job
    ft_config = config['finetuning']
    model = ft_config['base_model']
    suffix = ft_config['suffix']
    
    logger.info(f"Preparing to fine-tune base model: {model}")
    
    if args.dry_run:
        logger.info(f"[DRY RUN] Would start fine-tuning job with model={model}, suffix={suffix}")
        return 0
        
    try:
        hyperparams = {k: v for k, v in ft_config['hyperparameters'].items() if v != 'auto'}
        
        job_params = {
            "training_file": file_id,
            "model": model,
            "suffix": suffix
        }
        
        if hyperparams:
            job_params["hyperparameters"] = hyperparams
            
        job = client.fine_tuning.jobs.create(**job_params)
        
        job_id = job.id
        logger.info(f"Fine-tuning job started: {job_id}")
        
        # Save initial job details
        with open(job_output_path, 'w') as f:
            f.write(job.model_dump_json(indent=2))
            
    except Exception as e:
        logger.error(f"Failed to create fine-tuning job: {e}")
        return 1
        
    # 4. Monitor Job
    final_job = wait_for_job_completion(client, job_id, config['finetuning']['poll_interval_seconds'])
    
    # Save final job details
    with open(job_output_path, 'w') as f:
        f.write(final_job.model_dump_json(indent=2))
        
    if final_job.status == 'succeeded':
        fine_tuned_model = final_job.fine_tuned_model
        logger.info(f"Job succeeded! Fine-tuned model: {fine_tuned_model}")
        
        # Save model ID for chatbot
        output_data = {
            "model_id": fine_tuned_model,
            "job_id": job_id,
            "base_model": model,
            "created_at": final_job.created_at
        }
        
        # Ensure directory exists
        model_output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(model_output_path, 'w') as f:
            json.dump(output_data, f, indent=2)
            
        logger.info(f"Model info saved to {model_output_path}")
    else:
        logger.error(f"Fine-tuning job failed with status: {final_job.status}")
        if final_job.error:
            logger.error(f"Error: {final_job.error}")
        return 1
        
    return 0


if __name__ == '__main__':
    sys.exit(main())
