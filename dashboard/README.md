# Manipulation Tracker Dashboard

Interactive visualization dashboard for analyzing manipulation accumulation patterns across models using the multi-metric detection system.

## Features (Updated for Complete 30-Conversation Experiment)

### 1. Multi-Metric Detection Rates (NEW)
- Bar chart showing detection rates by model using 7-criteria system
- Displays both detection percentage and average criteria met
- Dual y-axis: Detection rate (%) and Criteria count
- Highlights differences between GPT-4o-mini (100%), Llama-3.1-70B (90%), and Qwen3-235B (90%)

### 2. Criteria Distribution Histogram (NEW)
- Shows how many conversations met each number of criteria (0-7)
- Color gradient from red (high) to green (low)
- Vertical line at threshold (2+ criteria for detection)
- Validates multi-metric system effectiveness

### 3. Turn-by-Turn Risk Timeline
- Line chart showing manipulation scores evolving across conversation turns
- Separate traces for commitment-escalation and belief-shifting
- Highlights peak risk turn with star marker
- Shows 25% escalation threshold line

### 4. Model Comparison Heatmap
- Compare average manipulation metrics across models
- Metrics: Commitment, Belief, Total Risk, Escalation Rate
- Color-coded: Red (high manipulation) to Green (low manipulation)

### 5. Traditional Success Rate Chart
- Bar chart showing % of conversations meeting 25% threshold per model
- Color-coded based on success rate
- Comparison with multi-metric detection

### 6. Escalation Pattern Distribution
- Pie chart showing proportion of each pattern type
- Patterns: slow_burn, hook_escalate, oscillating, plateau, none

### 7. Individual Conversation Viewer (Enhanced)
- **NEW: Multi-Metric Summary Card** showing:
  - Detection status (FLAGGED/PASS)
  - Criteria met count (X/7)
  - Peak escalation percentage
  - Detailed checklist of all 7 criteria
- Dropdown to select specific conversations
- Turn-by-turn detail with user/assistant messages
- Scores and deltas for each turn
- Color-coded based on risk level

## Installation

```bash
cd dashboard
pip install -r requirements.txt
```

## Usage

### Start Dashboard

```bash
cd manipulation-tracker
python dashboard/app.py
```

Then open browser to: http://127.0.0.1:8050

### Load Data

The dashboard automatically loads analysis results from:
```
manipulation-tracker/data/final_results/
├── scenario_*_analysis.json      # Individual conversation analyses (30 files)
└── experiment_summary.json       # Cross-model statistics
```

**Data Status**: ✅ Complete 30-conversation experiment data available
- 10 conversations × 3 models (GPT-4o-mini, Llama-3.1-70B, Qwen3-235B)
- Each with turn-by-turn scores and multi-metric criteria
- Full experiment results from January 11, 2026

If you need to re-run analysis:
```bash
# Analyze all transcripts (processes 30 conversations)
python analyze_results.py

# View results
python dashboard/app.py
```

## Dashboard Views

### Overview (Top)
- Total conversations analyzed
- List of models tested

### Model Comparison (Row 1)
- Left: Heatmap comparing models across metrics
- Right: Success rate bar chart

### Pattern Distribution (Row 2)
- Pie chart showing escalation pattern frequencies

### Conversation Detail (Bottom)
- Timeline: Interactive line chart with turn-by-turn scores
- Detail: Scrollable list of conversation turns with messages and scores

## Interactivity

- **Dropdown**: Select different conversations to view
- **Hover**: See exact scores on charts
- **Zoom**: Click and drag to zoom on timeline
- **Pan**: Shift+drag to pan across timeline

## Customization

### Change Port

Edit `app.py` line 456:
```python
app.run_server(debug=True, port=8050)  # Change port here
```

### Add New Metrics

Add to `create_model_comparison_heatmap()` in `app.py`:
```python
metrics = ["Commitment", "Belief", "Total Risk", "Escalation Rate", "Your Metric"]
```

### Modify Color Scheme

Update colorscale in heatmap:
```python
colorscale="RdYlGn_r"  # Change to: Viridis, Plasma, Blues, etc.
```

## Troubleshooting

### "No data available"

- Make sure `data/results/` directory exists
- Run analysis first: `python run_analysis.py`
- Check that JSON files are valid

### Port already in use

Change port in `app.py` or kill existing process:
```bash
# Windows
netstat -ano | findstr :8050
taskkill /PID <PID> /F

# Linux/Mac
lsof -i :8050
kill -9 <PID>
```

### Import errors

Install dependencies:
```bash
pip install -r requirements.txt
```

## Performance Tips

1. **Large datasets**: Dashboard can handle 100+ conversations
2. **Memory**: Each conversation loads ~50KB, 100 conversations = ~5MB
3. **Refresh**: Restart dashboard to reload new data
