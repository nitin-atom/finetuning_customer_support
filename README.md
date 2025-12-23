# Atom Helpdesk Fine-Tuning Dataset Generator

Generate high-quality synthetic Q&A training data from Atom.com helpdesk articles for fine-tuning an OpenAI customer support chatbot.

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run the full pipeline
python scripts/01_scrape.py          # Scrape all articles
python scripts/02_generate_questions.py   # Generate questions (batch API)
python scripts/03_generate_answers.py     # Generate answers (batch API)
python scripts/04_format_jsonl.py    # Format as JSONL
python scripts/05_quality_check.py   # Validate and finalize
```

## Pipeline Phases

### Phase 1: Scraping
Extracts all helpdesk articles from `https://helpdesk.atom.com/en/`

```bash
python scripts/01_scrape.py --limit 5    # Test with 5 articles
python scripts/01_scrape.py              # Full scrape (~242 articles)
python scripts/01_scrape.py --resume     # Resume from checkpoint
```

### Phase 2: Question Generation
Generates 3-5 diverse questions per article using GPT-4o Batch API.

```bash
python scripts/02_generate_questions.py --sync --limit 5  # Test synchronously
python scripts/02_generate_questions.py                   # Full batch run
```

### Phase 3: Answer Generation
Generates grounded answers for each question using GPT-4o Batch API.

```bash
python scripts/03_generate_answers.py --sync --limit 5  # Test synchronously
python scripts/03_generate_answers.py                   # Full batch run
```

### Phase 4: Formatting
Converts Q&A pairs to OpenAI fine-tuning JSONL format with collection-specific system prompts.

### Phase 5: Quality Assurance
Validates JSONL, checks content lengths, deduplicates, and generates quality report.

## Output Files

- `data/output/training_data.jsonl` - Training data (before QA)
- `data/output/final_training_data.jsonl` - Final validated training data
- `data/output/metadata.json` - Dataset statistics
- `data/output/quality_report.json` - Validation results

## Configuration

Edit `config/config.yaml` to adjust:
- OpenAI model and API settings
- Scraping rate limits
- Question/answer generation parameters
- Validation thresholds
- System prompts by collection
