"""
Validators for Q&A data quality checks.
"""

import re
import json
import logging
from typing import Optional
from difflib import SequenceMatcher

logger = logging.getLogger(__name__)


def validate_json_structure(line: str) -> tuple[bool, Optional[str]]:
    """Validate a JSONL line has correct structure."""
    try:
        data = json.loads(line)
    except json.JSONDecodeError as e:
        return False, f"Invalid JSON: {e}"
    
    # Check required fields
    if 'messages' not in data:
        return False, "Missing 'messages' field"
    
    messages = data['messages']
    if not isinstance(messages, list) or len(messages) != 3:
        return False, "Messages should be a list of 3 items (system, user, assistant)"
    
    # Check role sequence
    expected_roles = ['system', 'user', 'assistant']
    for i, (msg, expected_role) in enumerate(zip(messages, expected_roles)):
        if msg.get('role') != expected_role:
            return False, f"Message {i} should have role '{expected_role}', got '{msg.get('role')}'"
        if not msg.get('content'):
            return False, f"Message {i} has empty content"
    
    return True, None


def validate_content_length(question: str, answer: str, config: dict) -> tuple[bool, Optional[str]]:
    """Validate content lengths are within bounds."""
    min_q = config['validation']['min_question_chars']
    max_q = config['validation']['max_question_chars']
    min_a = config['validation']['min_answer_chars']
    max_a = config['validation']['max_answer_chars']
    
    if len(question) < min_q:
        return False, f"Question too short: {len(question)} < {min_q}"
    if len(question) > max_q:
        return False, f"Question too long: {len(question)} > {max_q}"
    if len(answer) < min_a:
        return False, f"Answer too short: {len(answer)} < {min_a}"
    if len(answer) > max_a:
        return False, f"Answer too long: {len(answer)} > {max_a}"
    
    return True, None


def check_exact_duplicates(qa_pairs: list[dict]) -> list[int]:
    """Find indices of exact duplicate Q&A pairs."""
    seen = {}
    duplicates = []
    
    for i, pair in enumerate(qa_pairs):
        key = (pair['question'].strip().lower(), pair['answer'].strip().lower())
        if key in seen:
            duplicates.append(i)
        else:
            seen[key] = i
    
    return duplicates


def check_near_duplicates(qa_pairs: list[dict], threshold: float = 0.95) -> list[tuple[int, int]]:
    """Find pairs of near-duplicate questions."""
    near_dupes = []
    
    for i in range(len(qa_pairs)):
        for j in range(i + 1, len(qa_pairs)):
            q1 = qa_pairs[i]['question'].strip().lower()
            q2 = qa_pairs[j]['question'].strip().lower()
            
            similarity = SequenceMatcher(None, q1, q2).ratio()
            if similarity >= threshold:
                near_dupes.append((i, j))
    
    return near_dupes


def validate_answer_grounding(answer: str, article_content: str) -> tuple[bool, list[str]]:
    """Basic check that answer content appears grounded in article.
    
    Returns (is_grounded, list of potentially hallucinated terms).
    """
    # Extract numbers and specific terms from answer
    answer_numbers = set(re.findall(r'\$?\d+(?:,\d{3})*(?:\.\d+)?%?', answer))
    article_numbers = set(re.findall(r'\$?\d+(?:,\d{3})*(?:\.\d+)?%?', article_content))
    
    # Check if numbers in answer exist in article
    ungrounded = []
    for num in answer_numbers:
        if num not in article_numbers and num not in ['1', '2', '3', '4', '5']:  # Allow simple counts
            ungrounded.append(num)
    
    # Check for hallucination markers
    hallucination_markers = [
        "I don't have information",
        "I cannot find",
        "not mentioned in",
        "According to the article",
        "based on the document",
        "the article states",
        "as mentioned in the article"
    ]
    
    for marker in hallucination_markers:
        if marker.lower() in answer.lower():
            ungrounded.append(f"marker: {marker}")
    
    is_grounded = len(ungrounded) == 0
    return is_grounded, ungrounded


def validate_question_format(question: str) -> tuple[bool, Optional[str]]:
    """Validate question is well-formed."""
    question = question.strip()
    
    if not question:
        return False, "Empty question"
    
    # Check it ends with question mark or is a statement-style query
    if not question.endswith('?') and not any(question.lower().startswith(w) for w in ['how', 'what', 'when', 'where', 'why', 'who', 'can', 'do', 'does', 'is', 'are', 'will', 'should']):
        # Allow brief statement-style queries like "Commission rates"
        if len(question) > 30:
            return False, "Long question should end with '?'"
    
    return True, None


def parse_questions_json(response: str) -> list[dict]:
    """Parse question generation response, handling potential formatting issues."""
    # Try to find JSON array in response
    response = response.strip()
    
    # Handle markdown code blocks
    if '```json' in response:
        match = re.search(r'```json\s*(.*?)\s*```', response, re.DOTALL)
        if match:
            response = match.group(1)
    elif '```' in response:
        match = re.search(r'```\s*(.*?)\s*```', response, re.DOTALL)
        if match:
            response = match.group(1)
    
    # Try to find JSON array
    start = response.find('[')
    end = response.rfind(']')
    
    if start != -1 and end != -1:
        try:
            questions = json.loads(response[start:end + 1])
            return questions
        except json.JSONDecodeError:
            pass
    
    # Try full response
    try:
        questions = json.loads(response)
        return questions
    except json.JSONDecodeError:
        logger.warning(f"Failed to parse questions JSON: {response[:200]}...")
        return []


def deduplicate_qa_pairs(qa_pairs: list[dict], config: dict) -> tuple[list[dict], dict]:
    """Remove duplicate Q&A pairs and return deduped list with stats."""
    threshold = config['validation']['similarity_threshold']
    
    # Remove exact duplicates
    exact_dupes = check_exact_duplicates(qa_pairs)
    
    # Create list without exact duplicates
    no_exact = [pair for i, pair in enumerate(qa_pairs) if i not in exact_dupes]
    
    # Find near duplicates
    near_dupes = check_near_duplicates(no_exact, threshold)
    
    # Remove second item from each near-duplicate pair
    near_dupe_indices = set(j for i, j in near_dupes)
    deduped = [pair for i, pair in enumerate(no_exact) if i not in near_dupe_indices]
    
    stats = {
        'original_count': len(qa_pairs),
        'exact_duplicates_removed': len(exact_dupes),
        'near_duplicates_removed': len(near_dupes),
        'final_count': len(deduped)
    }
    
    return deduped, stats
