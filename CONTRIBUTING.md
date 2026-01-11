# Contributing to Manipulation Accumulation Tracker

Thank you for your interest in contributing! This document provides guidelines for contributing to the project.

## Code of Conduct

This project is dedicated to providing a welcoming environment for all contributors. Please be respectful and constructive in all interactions.

## How to Contribute

### Reporting Issues

- Check if the issue already exists
- Provide clear description with steps to reproduce
- Include relevant logs, error messages, or screenshots
- Specify your environment (Python version, OS, etc.)

### Suggesting Enhancements

- Open an issue tagged as "enhancement"
- Clearly describe the proposed feature
- Explain why it would be valuable
- Consider providing example use cases

### Pull Requests

1. **Fork the repository** and create your branch from `main`

2. **Make your changes**:
   - Write clean, readable code
   - Follow existing code style
   - Add comments for complex logic
   - Update documentation as needed

3. **Test your changes**:
   - Ensure existing tests pass
   - Add tests for new features
   - Test with multiple models if applicable

4. **Commit your changes**:
   ```bash
   git add .
   git commit -m "Brief description of changes"
   ```

5. **Push to your fork** and submit a pull request

6. **PR Description should include**:
   - What changes were made
   - Why the changes were necessary
   - Any potential impacts or breaking changes
   - Testing performed

## Development Setup

```bash
# Clone your fork
git clone https://github.com/YOUR-USERNAME/manipulation-tracker
cd manipulation-tracker

# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install in development mode
pip install -e .
pip install -r requirements.txt

# Install pre-commit hooks (optional)
pip install pre-commit
pre-commit install
```

## Project Structure

```
src/
├── models.py              # Data structures
├── turn_scorer.py         # Scoring logic
├── cumulative_tracker.py  # Tracking logic
├── metrics_calculator.py  # Metrics computation
└── manipulation_judgment.py # BLOOM integration

dashboard/
├── app.py                 # Main dashboard app
└── components/            # Dashboard components

seeds/
├── behaviors.json         # Behavior definitions
└── scenarios/             # Test scenarios
```

## Areas for Contribution

### High Priority

1. **Additional Manipulation Categories**
   - Emotional manipulation
   - Social proof tactics
   - Authority exploitation
   - Scarcity/urgency creation

2. **Improved Pattern Detection**
   - Better algorithms for identifying oscillating patterns
   - Statistical significance testing
   - Anomaly detection

3. **Multilingual Support**
   - Non-English scenario generation
   - Cross-lingual evaluation
   - Language-specific manipulation tactics

### Medium Priority

4. **Dashboard Enhancements**
   - Export functionality
   - Comparative analysis views
   - Real-time monitoring mode

5. **Model Provider Support**
   - Additional API integrations
   - Local model support
   - Batch evaluation capabilities

6. **Documentation**
   - Tutorial notebooks
   - Video walkthroughs
   - Case studies

### Ideas Welcome

- Integration with other safety frameworks
- Automated report generation
- Longitudinal tracking across sessions
- User study tools

## Coding Standards

### Python Style

- Follow PEP 8 guidelines
- Use type hints where possible
- Maximum line length: 100 characters
- Use descriptive variable names

Example:
```python
def score_turn(
    self,
    conversation_history: List[Dict[str, str]],
    current_turn_idx: int,
    scenario_description: str = ""
) -> TurnScore:
    """
    Score a single conversation turn for manipulation.

    Args:
        conversation_history: Full conversation up to current turn
        current_turn_idx: Index of turn to score (0-based)
        scenario_description: Optional context about scenario

    Returns:
        TurnScore object with category scores and metadata
    """
    # Implementation...
```

### Commit Messages

- Use present tense ("Add feature" not "Added feature")
- Use imperative mood ("Move cursor to..." not "Moves cursor to...")
- First line max 50 characters
- Reference issues/PRs when relevant

Good examples:
```
Add emotional manipulation category
Fix turn indexing bug in scorer
Update dashboard with new metrics (#42)
```

## Testing

### Running Tests

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_turn_scorer.py

# Run with coverage
pytest --cov=src tests/
```

### Writing Tests

- Place tests in `tests/` directory
- Name test files `test_*.py`
- Name test functions `test_*`
- Use descriptive test names

Example:
```python
def test_turn_scorer_handles_empty_conversation():
    scorer = TurnScorer()
    with pytest.raises(ValueError):
        scorer.score_turn(conversation_history=[], current_turn_idx=0)
```

## Documentation

- Update README.md for user-facing changes
- Update docstrings for code changes
- Add examples for new features
- Keep CHANGELOG.md current

## Questions?

- Open an issue tagged "question"
- Check existing documentation
- Review closed issues for similar questions

## Attribution

Contributors will be acknowledged in:
- README.md contributors section
- Release notes
- Academic papers (for significant contributions)

Thank you for contributing to AI safety research!
