"""
OpenAI LLM client with batch API support for question and answer generation.
"""

import os
import json
import time
import logging
from pathlib import Path
from typing import Optional
from openai import OpenAI
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()


class LLMClient:
    """OpenAI client wrapper with batch API support."""
    
    def __init__(self, config: dict):
        self.config = config
        self.client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        self.model = config['openai']['model']
        self.max_retries = config['openai']['max_retries']
        self.retry_delay = config['openai']['retry_delay_seconds']
        self.batch_check_interval = config['openai']['batch_check_interval_seconds']
    
    def generate_single(self, prompt: str, temperature: float = 0.7, max_tokens: int = 1000) -> Optional[str]:
        """Generate a single completion (for testing/fallback)."""
        for attempt in range(self.max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=temperature,
                    max_tokens=max_tokens
                )
                return response.choices[0].message.content
            except Exception as e:
                logger.warning(f"Attempt {attempt + 1} failed: {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay * (2 ** attempt))
                else:
                    logger.error(f"Failed after {self.max_retries} attempts")
                    return None
        return None
    
    def create_batch_file(self, requests: list[dict], output_path: Path) -> Path:
        """Create a JSONL file for batch processing.
        
        Each request should have:
        - custom_id: Unique identifier for the request
        - prompt: The prompt to send
        - temperature: Optional temperature override
        - max_tokens: Optional max_tokens override
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            for req in requests:
                batch_request = {
                    "custom_id": req['custom_id'],
                    "method": "POST",
                    "url": "/v1/chat/completions",
                    "body": {
                        "model": self.model,
                        "messages": [{"role": "user", "content": req['prompt']}],
                        "temperature": req.get('temperature', 0.7),
                        "max_tokens": req.get('max_tokens', 1000)
                    }
                }
                f.write(json.dumps(batch_request) + '\n')
        
        logger.info(f"Created batch file with {len(requests)} requests: {output_path}")
        return output_path
    
    def upload_batch_file(self, file_path: Path) -> str:
        """Upload batch file to OpenAI and return file ID."""
        with open(file_path, 'rb') as f:
            file_response = self.client.files.create(
                file=f,
                purpose='batch'
            )
        logger.info(f"Uploaded batch file: {file_response.id}")
        return file_response.id
    
    def submit_batch(self, file_id: str, description: str = "Atom helpdesk batch") -> str:
        """Submit a batch job and return batch ID."""
        batch = self.client.batches.create(
            input_file_id=file_id,
            endpoint="/v1/chat/completions",
            completion_window="24h",
            metadata={"description": description}
        )
        logger.info(f"Submitted batch job: {batch.id}")
        return batch.id
    
    def wait_for_batch(self, batch_id: str, max_wait_hours: int = 24) -> dict:
        """Wait for batch to complete and return status."""
        max_wait_seconds = max_wait_hours * 3600
        start_time = time.time()
        
        while True:
            batch = self.client.batches.retrieve(batch_id)
            status = batch.status
            
            logger.info(f"Batch {batch_id} status: {status}")
            
            if status == 'completed':
                return {
                    'status': 'completed',
                    'output_file_id': batch.output_file_id,
                    'request_counts': {
                        'total': batch.request_counts.total,
                        'completed': batch.request_counts.completed,
                        'failed': batch.request_counts.failed
                    }
                }
            elif status in ['failed', 'expired', 'cancelled']:
                return {
                    'status': status,
                    'error': getattr(batch, 'errors', None)
                }
            
            # Check timeout
            if time.time() - start_time > max_wait_seconds:
                logger.warning(f"Batch {batch_id} timed out after {max_wait_hours} hours")
                return {'status': 'timeout'}
            
            # Wait before checking again
            time.sleep(self.batch_check_interval)
    
    def get_batch_results(self, output_file_id: str) -> list[dict]:
        """Download and parse batch results."""
        content = self.client.files.content(output_file_id)
        results = []
        
        for line in content.text.strip().split('\n'):
            if line:
                result = json.loads(line)
                custom_id = result['custom_id']
                
                if result.get('error'):
                    results.append({
                        'custom_id': custom_id,
                        'error': result['error'],
                        'content': None
                    })
                else:
                    content = result['response']['body']['choices'][0]['message']['content']
                    results.append({
                        'custom_id': custom_id,
                        'error': None,
                        'content': content
                    })
        
        logger.info(f"Retrieved {len(results)} batch results")
        return results
    
    def run_batch(self, requests: list[dict], batch_file_path: Path, description: str = "Batch job") -> list[dict]:
        """Full batch workflow: create file, upload, submit, wait, get results."""
        # Create batch file
        self.create_batch_file(requests, batch_file_path)
        
        # Upload file
        file_id = self.upload_batch_file(batch_file_path)
        
        # Submit batch
        batch_id = self.submit_batch(file_id, description)
        
        # Wait for completion
        result = self.wait_for_batch(batch_id)
        
        if result['status'] != 'completed':
            logger.error(f"Batch failed with status: {result['status']}")
            return []
        
        # Get results
        return self.get_batch_results(result['output_file_id'])


def load_prompt(prompt_path: Path) -> str:
    """Load a prompt template from file."""
    with open(prompt_path, 'r') as f:
        return f.read()


def format_prompt(template: str, **kwargs) -> str:
    """Format a prompt template with variables."""
    return template.format(**kwargs)
