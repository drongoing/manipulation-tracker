#!/usr/bin/env python3
"""
Manipulation Accumulation Tracker - Interactive Dashboard

Visualizes turn-by-turn manipulation patterns across models.
"""

import json
from pathlib import Path
from typing import List, Dict
import dash
from dash import dcc, html, Input, Output
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd


def shorten_model_name(model_full_name: str) -> str:
    """
    Convert full model path to short display name.

    Examples:
        openai/gpt-4o-mini -> GPT-4o-mini
        together_ai/meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo -> Llama-3.1-70B
        together_ai/Qwen/Qwen3-235B-A22B-Instruct-2507-tput -> Qwen3-235B
    """
    if "gpt-4o-mini" in model_full_name:
        return "GPT-4o-mini"
    elif "Llama-3.1-70B" in model_full_name:
        return "Llama-3.1-70B"
    elif "Qwen3-235B" in model_full_name:
        return "Qwen3-235B"
    else:
        # Fallback: take last part after /
        return model_full_name.split("/")[-1]


class DashboardData:
    """Loads and prepares data for visualization."""

    def __init__(self, results_dir: Path):
        """
        Initialize dashboard data loader.

        Args:
            results_dir: Directory containing analysis JSON files
        """
        self.results_dir = results_dir
        self.analyses = []
        self.aggregate = None
        self.load_data()

    def load_data(self):
        """Load all analysis results from directory."""
        if not self.results_dir.exists():
            print(f"Warning: Results directory not found: {self.results_dir}")
            return

        # Load individual analyses - support multiple naming patterns
        analysis_files = list(self.results_dir.glob("analysis_*.json"))
        analysis_files.extend(self.results_dir.glob("scenario_*_analysis.json"))
        analysis_files = sorted(set(analysis_files))

        for filepath in analysis_files:
            if filepath.name in ["aggregate_results.json", "experiment_summary.json"]:
                continue

            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.analyses.append(data)
            except Exception as e:
                print(f"Error loading {filepath}: {e}")

        # Load aggregate results if available
        aggregate_path = self.results_dir / "aggregate_results.json"
        if not aggregate_path.exists():
            aggregate_path = self.results_dir / "experiment_summary.json"

        if aggregate_path.exists():
            with open(aggregate_path, 'r', encoding='utf-8') as f:
                self.aggregate = json.load(f)

        print(f"Loaded {len(self.analyses)} analyses")

    def get_scenario_ids(self) -> List[str]:
        """Get list of all scenario IDs."""
        return [a["scenario_id"] for a in self.analyses]

    def get_models(self) -> List[str]:
        """Get list of unique model names."""
        return sorted(list(set(a["model_tested"] for a in self.analyses)))

    def get_analysis(self, scenario_id: str) -> Dict:
        """Get analysis for specific scenario."""
        for analysis in self.analyses:
            if analysis["scenario_id"] == scenario_id:
                return analysis
        return None


def create_timeline_chart(analysis: Dict) -> go.Figure:
    """
    Create turn-by-turn risk timeline chart.

    Shows how manipulation scores evolve across conversation turns.
    """
    if not analysis:
        return go.Figure()

    turns = analysis["turns"]
    turn_numbers = [t["turn_number"] for t in turns]

    # Extract category scores
    commitment_scores = [t["category_scores"].get("commitment-escalation", 0) for t in turns]
    belief_scores = [t["category_scores"].get("belief-shifting", 0) for t in turns]
    total_scores = [t["total_score"] for t in turns]

    # Create figure with secondary y-axis
    fig = make_subplots(
        rows=1, cols=1,
        subplot_titles=(f"Manipulation Escalation: {analysis['scenario_id']}",)
    )

    # Add traces
    fig.add_trace(
        go.Scatter(
            x=turn_numbers,
            y=commitment_scores,
            name="Commitment Escalation",
            mode="lines+markers",
            line=dict(color="rgb(255, 127, 14)", width=2),
            marker=dict(size=8)
        )
    )

    fig.add_trace(
        go.Scatter(
            x=turn_numbers,
            y=belief_scores,
            name="Belief Shifting",
            mode="lines+markers",
            line=dict(color="rgb(44, 160, 44)", width=2),
            marker=dict(size=8)
        )
    )

    fig.add_trace(
        go.Scatter(
            x=turn_numbers,
            y=total_scores,
            name="Total Risk",
            mode="lines+markers",
            line=dict(color="rgb(214, 39, 40)", width=3, dash="dash"),
            marker=dict(size=10, symbol="diamond")
        )
    )

    # Mark peak risk turn
    peak_turn = analysis["peak_risk_turn"]
    peak_risk = analysis["peak_risk"]

    fig.add_trace(
        go.Scatter(
            x=[peak_turn],
            y=[peak_risk],
            name="Peak Risk",
            mode="markers",
            marker=dict(size=20, color="red", symbol="star", line=dict(width=2, color="darkred"))
        )
    )

    # Add threshold line (25% escalation)
    initial_risk = turns[0]["total_score"]
    threshold_risk = initial_risk * 1.25

    fig.add_hline(
        y=threshold_risk,
        line_dash="dot",
        line_color="gray",
        annotation_text="25% Threshold",
        annotation_position="right"
    )

    # Layout
    fig.update_layout(
        xaxis_title="Turn Number",
        yaxis_title="Manipulation Score (1-10 per category)",
        yaxis_range=[0, 20],
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height=500
    )

    return fig


def create_model_comparison_chart(data: DashboardData) -> go.Figure:
    """
    Create model comparison clustered column chart.

    Compares Commitment Escalation and Belief Shifting across models.
    """
    if not data.analyses:
        return go.Figure()

    # Prepare data
    models = data.get_models()
    model_short_names = [shorten_model_name(m) for m in models]

    commitment_scores = []
    belief_scores = []

    for model in models:
        model_analyses = [a for a in data.analyses if a["model_tested"] == model]

        if not model_analyses:
            commitment_scores.append(0)
            belief_scores.append(0)
            continue

        # Average commitment escalation
        avg_commitment = sum(
            sum(t["category_scores"].get("commitment-escalation", 0) for t in a["turns"]) / len(a["turns"])
            for a in model_analyses
        ) / len(model_analyses)

        # Average belief shifting
        avg_belief = sum(
            sum(t["category_scores"].get("belief-shifting", 0) for t in a["turns"]) / len(a["turns"])
            for a in model_analyses
        ) / len(model_analyses)

        commitment_scores.append(avg_commitment)
        belief_scores.append(avg_belief)

    # Create clustered column chart
    fig = go.Figure()

    fig.add_trace(go.Bar(
        name='Commitment Escalation',
        x=model_short_names,
        y=commitment_scores,
        text=[f"{val:.1f}" for val in commitment_scores],
        textposition="outside",
        marker=dict(color='rgb(255, 127, 14)')
    ))

    fig.add_trace(go.Bar(
        name='Belief Shifting',
        x=model_short_names,
        y=belief_scores,
        text=[f"{val:.1f}" for val in belief_scores],
        textposition="outside",
        marker=dict(color='rgb(44, 160, 44)')
    ))

    fig.update_layout(
        title="Model Comparison: Average Manipulation Metrics",
        xaxis_title="Model",
        yaxis_title="Average Score",
        yaxis_range=[0, max(max(commitment_scores), max(belief_scores)) * 1.2],
        barmode='group',
        height=400,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )

    return fig


def create_pattern_distribution_chart(data: DashboardData) -> go.Figure:
    """
    Create escalation pattern distribution pie chart.

    Shows proportion of each escalation pattern across all conversations.
    """
    if not data.analyses:
        return go.Figure()

    # Count pattern occurrences
    pattern_counts = {}
    for analysis in data.analyses:
        pattern = analysis["pattern_detected"]
        pattern_counts[pattern] = pattern_counts.get(pattern, 0) + 1

    # Create pie chart
    fig = go.Figure(data=[go.Pie(
        labels=list(pattern_counts.keys()),
        values=list(pattern_counts.values()),
        hole=0.3,
        marker=dict(colors=px.colors.qualitative.Set3)
    )])

    fig.update_layout(
        title="Escalation Pattern Distribution",
        height=400
    )

    return fig


def create_success_rate_chart(data: DashboardData) -> go.Figure:
    """
    Create success rate bar chart by model.

    Shows percentage of conversations meeting 25% escalation threshold.
    """
    if not data.analyses:
        return go.Figure()

    models = data.get_models()
    model_short_names = [shorten_model_name(m) for m in models]
    success_rates = []

    for model in models:
        model_analyses = [a for a in data.analyses if a["model_tested"] == model]
        if model_analyses:
            successes = sum(1 for a in model_analyses if a["success_threshold_met"])
            rate = successes / len(model_analyses) * 100
            success_rates.append(rate)
        else:
            success_rates.append(0)

    # Create bar chart
    fig = go.Figure(data=[
        go.Bar(
            x=model_short_names,
            y=success_rates,
            text=[f"{rate:.1f}%" for rate in success_rates],
            textposition="outside",
            marker=dict(color=success_rates, colorscale="RdYlGn", cmin=0, cmax=100)
        )
    ])

    # Add threshold line at 50%
    fig.add_hline(y=50, line_dash="dash", line_color="gray", annotation_text="50% baseline")

    fig.update_layout(
        title="Success Rate by Model (25% Escalation Threshold)",
        xaxis_title="Model",
        yaxis_title="Success Rate (%)",
        yaxis_range=[0, 110],
        height=400
    )

    return fig


def create_detection_rate_chart(data: DashboardData) -> go.Figure:
    """
    Create detection rate bar chart by model using multi-metric system.

    Shows percentage of conversations flagged by multi-metric criteria (4+ criteria required).
    """
    if not data.analyses:
        return go.Figure()

    models = data.get_models()
    model_short_names = [shorten_model_name(m) for m in models]
    detection_rates = []

    for model in models:
        model_analyses = [a for a in data.analyses if a["model_tested"] == model]
        if model_analyses:
            # Count conversations where manipulation_detected is True
            detected = sum(1 for a in model_analyses
                          if a.get("metrics", {}).get("manipulation_detected", False))
            rate = detected / len(model_analyses) * 100
            detection_rates.append(rate)
        else:
            detection_rates.append(0)

    # Create bar chart
    fig = go.Figure(data=[
        go.Bar(
            x=model_short_names,
            y=detection_rates,
            text=[f"{rate:.1f}%" for rate in detection_rates],
            textposition="outside",
            marker=dict(color='rgb(214, 39, 40)')
        )
    ])

    fig.update_layout(
        title="Multi-Metric Detection Rates by Model",
        xaxis_title="Model",
        yaxis_title="Detection Rate (%)",
        yaxis_range=[0, 110],
        height=400
    )

    return fig


def create_avg_criteria_chart(data: DashboardData) -> go.Figure:
    """
    Create average criteria met bar chart by model.

    Shows average number of detection criteria met (out of 7) for each model.
    """
    if not data.analyses:
        return go.Figure()

    models = data.get_models()
    model_short_names = [shorten_model_name(m) for m in models]
    avg_criteria = []

    for model in models:
        model_analyses = [a for a in data.analyses if a["model_tested"] == model]
        if model_analyses:
            # Average criteria met
            avg_crit = sum(a.get("metrics", {}).get("criteria_met_count", 0)
                          for a in model_analyses) / len(model_analyses)
            avg_criteria.append(avg_crit)
        else:
            avg_criteria.append(0)

    # Create bar chart
    fig = go.Figure(data=[
        go.Bar(
            x=model_short_names,
            y=avg_criteria,
            text=[f"{crit:.1f}/7" for crit in avg_criteria],
            textposition="outside",
            marker=dict(color='rgb(31, 119, 180)')
        )
    ])

    fig.update_layout(
        title="Average Criteria Met by Model",
        xaxis_title="Model",
        yaxis_title="Average Criteria Met (out of 7)",
        yaxis_range=[0, 8],
        height=400
    )

    return fig


def create_all_conversations_summary_chart(data: DashboardData) -> go.Figure:
    """
    Create summary chart showing turn-by-turn manipulation score trends across all conversations.

    Displays average commitment, belief, and total scores at each turn position across all 30 conversations.
    Also marks the peak manipulation turn.
    """
    if not data.analyses:
        return go.Figure()

    # Initialize dictionaries for each score type by turn position
    commitment_scores = {}
    belief_scores = {}
    total_scores = {}

    for analysis in data.analyses:
        for turn in analysis["turns"]:
            turn_num = turn["turn_number"]

            # Initialize lists if needed
            if turn_num not in commitment_scores:
                commitment_scores[turn_num] = []
                belief_scores[turn_num] = []
                total_scores[turn_num] = []

            # Collect scores
            commitment_scores[turn_num].append(turn["category_scores"].get("commitment-escalation", 0))
            belief_scores[turn_num].append(turn["category_scores"].get("belief-shifting", 0))
            total_scores[turn_num].append(turn["total_score"])

    # Calculate averages for each turn position
    turn_numbers = sorted(total_scores.keys())
    avg_commitment = [sum(commitment_scores[t]) / len(commitment_scores[t]) if commitment_scores[t] else 0 for t in turn_numbers]
    avg_belief = [sum(belief_scores[t]) / len(belief_scores[t]) if belief_scores[t] else 0 for t in turn_numbers]
    avg_total = [sum(total_scores[t]) / len(total_scores[t]) if total_scores[t] else 0 for t in turn_numbers]

    # Find peak turn (highest average total score)
    peak_turn_idx = avg_total.index(max(avg_total))
    peak_turn_num = turn_numbers[peak_turn_idx]
    peak_score = avg_total[peak_turn_idx]

    # Create line chart
    fig = go.Figure()

    # Add Commitment Escalation trace
    fig.add_trace(go.Scatter(
        x=turn_numbers,
        y=avg_commitment,
        mode='lines+markers',
        name='Commitment Escalation',
        line=dict(color='rgb(255, 127, 14)', width=2),
        marker=dict(size=6)
    ))

    # Add Belief Shifting trace
    fig.add_trace(go.Scatter(
        x=turn_numbers,
        y=avg_belief,
        mode='lines+markers',
        name='Belief Shifting',
        line=dict(color='rgb(44, 160, 44)', width=2),
        marker=dict(size=6)
    ))

    # Add Total Risk trace
    fig.add_trace(go.Scatter(
        x=turn_numbers,
        y=avg_total,
        mode='lines+markers',
        name='Total Risk',
        line=dict(color='rgb(214, 39, 40)', width=3, dash='dash'),
        marker=dict(size=8, symbol='diamond')
    ))

    # Mark peak turn
    fig.add_trace(go.Scatter(
        x=[peak_turn_num],
        y=[peak_score],
        mode='markers',
        name='Peak Manipulation Turn',
        marker=dict(size=20, color='red', symbol='star', line=dict(width=2, color='darkred'))
    ))

    fig.update_layout(
        title="Turn-by-Turn Manipulation Trends Across All 30 Conversations",
        xaxis_title="Turn Number",
        yaxis_title="Average Manipulation Score",
        yaxis_range=[0, 20],
        hovermode="x unified",
        height=400,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )

    return fig


def create_criteria_distribution_chart(data: DashboardData) -> go.Figure:
    """
    Create histogram of criteria met distribution.

    Shows how many conversations met each number of criteria (0-7).
    """
    if not data.analyses:
        return go.Figure()

    # Count conversations by criteria met
    criteria_counts = {i: 0 for i in range(8)}

    for analysis in data.analyses:
        criteria_met = analysis.get("metrics", {}).get("criteria_met_count", 0)
        criteria_counts[criteria_met] = criteria_counts.get(criteria_met, 0) + 1

    fig = go.Figure(data=[
        go.Bar(
            x=list(criteria_counts.keys()),
            y=list(criteria_counts.values()),
            text=list(criteria_counts.values()),
            textposition="outside",
            marker=dict(
                color=list(criteria_counts.keys()),
                colorscale='RdYlGn_r',
                showscale=True,
                colorbar=dict(title="Criteria Met")
            )
        )
    ])

    fig.add_vline(x=4, line_dash="dash", line_color="red",
                  annotation_text="Detection Threshold (4+)",
                  annotation_position="top right")

    fig.update_layout(
        title="Distribution of Criteria Met (Multi-Metric System)",
        xaxis_title="Number of Criteria Met (out of 7)",
        yaxis_title="Number of Conversations",
        height=400,
        xaxis=dict(tickmode='linear', tick0=0, dtick=1)
    )

    return fig


def create_conversation_detail(analysis: Dict) -> html.Div:
    """
    Create detailed conversation viewer.

    Shows turn-by-turn messages and scores plus multi-metric criteria.
    """
    if not analysis:
        return html.Div("No conversation selected")

    turns = analysis["turns"]

    # Add metrics summary card at the top
    metrics = analysis.get("metrics", {})
    manipulation_detected = metrics.get("manipulation_detected", False)
    criteria_met_count = metrics.get("criteria_met_count", 0)
    criteria_details = metrics.get("criteria_details", {})
    peak_escalation = metrics.get("peak_escalation_rate", 0)

    metrics_card = html.Div([
        html.H4(f"Multi-Metric Analysis Results", style={"marginBottom": "15px", "color": "#667eea"}),
        html.Div([
            html.Span("Status: ", style={"fontWeight": "bold"}),
            html.Span("FLAGGED", style={"color": "red", "fontWeight": "bold", "fontSize": "18px"}) if manipulation_detected
            else html.Span("PASS", style={"color": "green", "fontWeight": "bold", "fontSize": "18px"})
        ], style={"marginBottom": "10px"}),
        html.Div([
            html.Span(f"Criteria Met: {criteria_met_count}/7", style={"marginRight": "20px", "fontWeight": "bold"}),
            html.Span(f"Peak Escalation: {peak_escalation:.1f}%", style={"fontWeight": "bold"})
        ], style={"marginBottom": "15px"}),
        html.Div([
            html.H5("Criteria Details:", style={"marginBottom": "10px"}),
            html.Ul([
                html.Li([
                    html.Span("✓ " if val else "✗ ", style={"fontWeight": "bold", "color": "green" if val else "red"}),
                    html.Span(key.replace("_", " ").title())
                ]) for key, val in criteria_details.items()
            ], style={"listStyle": "none", "padding": "0"})
        ])
    ], style={
        "marginBottom": "30px",
        "padding": "20px",
        "border": "2px solid " + ("red" if manipulation_detected else "green"),
        "borderRadius": "10px",
        "backgroundColor": "#f9f9f9"
    })

    conversation_cards = [metrics_card]

    for turn in turns:
        turn_num = turn["turn_number"]
        user_msg = turn["raw_user_message"]
        assistant_msg = turn["raw_assistant_message"]
        commit_score = turn["category_scores"].get("commitment-escalation", 0)
        belief_score = turn["category_scores"].get("belief-shifting", 0)
        total_score = turn["total_score"]
        delta = turn["delta_from_previous"]

        # Determine color based on total score
        if total_score < 5:
            score_color = "green"
        elif total_score < 10:
            score_color = "orange"
        else:
            score_color = "red"

        card = html.Div([
            html.H5(f"Turn {turn_num}", style={"marginBottom": "10px"}),

            html.Div([
                html.Strong("User: "),
                html.Span(user_msg[:200] + ("..." if len(user_msg) > 200 else ""))
            ], style={"marginBottom": "10px", "padding": "10px", "backgroundColor": "#f0f0f0", "borderRadius": "5px"}),

            html.Div([
                html.Strong("Assistant: "),
                html.Span(assistant_msg[:200] + ("..." if len(assistant_msg) > 200 else ""))
            ], style={"marginBottom": "10px", "padding": "10px", "backgroundColor": "#e8f4f8", "borderRadius": "5px"}),

            html.Div([
                html.Span(f"Commitment: {commit_score:.1f}/10 | ", style={"marginRight": "10px"}),
                html.Span(f"Belief: {belief_score:.1f}/10 | ", style={"marginRight": "10px"}),
                html.Span(f"Total: {total_score:.1f}/20 ", style={"fontWeight": "bold", "color": score_color}),
                html.Span(f"({delta:+.1f})", style={"color": "gray"})
            ], style={"fontSize": "14px", "marginBottom": "5px"}),

        ], style={"marginBottom": "30px", "padding": "15px", "border": "1px solid #ddd", "borderRadius": "8px"})

        conversation_cards.append(card)

    return html.Div(conversation_cards, style={"maxHeight": "600px", "overflowY": "scroll"})


# Initialize Dash app
app = dash.Dash(__name__, title="Manipulation Tracker Dashboard")

# Load data
# Handle both direct execution and module import
try:
    DATA_DIR = Path(__file__).parent.parent / "data" / "final_results"
    # Fallback to old location if new one doesn't exist
    if not DATA_DIR.exists():
        DATA_DIR = Path(__file__).parent.parent / "data" / "results"
except NameError:
    # Fallback if __file__ is not defined
    DATA_DIR = Path.cwd() / "data" / "final_results"
    if not DATA_DIR.exists():
        DATA_DIR = Path.cwd() / "data" / "results"

data = DashboardData(DATA_DIR)

# Define layout
app.layout = html.Div([
    html.H1("Manipulation Accumulation Tracker - Complete Experiment Results",
            style={"textAlign": "center", "marginBottom": "20px", "color": "#667eea"}),

    html.Div([
        html.P(f"Total Conversations: {len(data.analyses)} | Models: {', '.join(data.get_models())}",
               style={"textAlign": "center", "fontSize": "18px", "color": "#666", "fontWeight": "bold"}),
        html.P("Multi-Metric Detection System: 7 Criteria, 4+ Required for Flagging",
               style={"textAlign": "center", "fontSize": "14px", "color": "#999", "marginTop": "5px"})
    ]),

    html.Hr(),

    # Row 1: Detection Rates and Average Criteria
    html.H3("Multi-Metric Detection Results", style={"marginTop": "20px", "marginBottom": "10px"}),
    html.Div([
        html.Div([
            dcc.Graph(id="detection-rate-chart", figure=create_detection_rate_chart(data)),
            html.P("Percentage of conversations flagged by the multi-metric system (4+ criteria required for detection)",
                   style={"textAlign": "center", "fontSize": "12px", "color": "#666", "fontStyle": "italic", "marginTop": "10px"})
        ], style={"width": "50%", "display": "inline-block", "padding": "10px", "verticalAlign": "top"}),

        html.Div([
            dcc.Graph(id="avg-criteria-chart", figure=create_avg_criteria_chart(data)),
            html.P("Average number of detection criteria met across all conversations for each model (out of 7 total)",
                   style={"textAlign": "center", "fontSize": "12px", "color": "#666", "fontStyle": "italic", "marginTop": "10px"})
        ], style={"width": "50%", "display": "inline-block", "padding": "10px", "verticalAlign": "top"}),
    ]),

    html.Hr(),

    # Row 2: Criteria Distribution
    html.H3("Detection Criteria Distribution", style={"marginTop": "20px", "marginBottom": "10px"}),
    html.Div([
        html.Div([
            dcc.Graph(id="criteria-distribution", figure=create_criteria_distribution_chart(data)),
            html.P("Distribution showing how many conversations met each number of criteria (threshold: 4+ for flagging)",
                   style={"textAlign": "center", "fontSize": "12px", "color": "#666", "fontStyle": "italic", "marginTop": "10px"})
        ], style={"width": "70%", "margin": "auto", "padding": "10px"}),
    ]),

    html.Hr(),

    # Row 3: Model Comparison
    html.H3("Model Performance Comparison", style={"marginTop": "20px", "marginBottom": "10px"}),
    html.Div([
        html.Div([
            dcc.Graph(id="model-comparison-chart", figure=create_model_comparison_chart(data)),
            html.P("Average scores for Commitment Escalation and Belief Shifting across models",
                   style={"textAlign": "center", "fontSize": "12px", "color": "#666", "fontStyle": "italic", "marginTop": "10px"})
        ], style={"width": "80%", "margin": "auto", "padding": "10px"}),
    ]),

    html.Hr(),

    # Row 4: Pattern Distribution
    html.H3("Escalation Pattern Analysis", style={"marginTop": "20px", "marginBottom": "10px"}),
    html.Div([
        html.Div([
            dcc.Graph(id="pattern-pie", figure=create_pattern_distribution_chart(data)),
            html.P("Distribution of escalation patterns detected across all conversations",
                   style={"textAlign": "center", "fontSize": "12px", "color": "#666", "fontStyle": "italic", "marginTop": "10px"})
        ], style={"width": "50%", "margin": "auto", "padding": "10px"}),
    ]),

    html.Hr(),

    # Row 5: All Conversations Summary
    html.H2("Summary of All 30 Conversations", style={"marginTop": "30px", "marginBottom": "10px"}),
    html.Div([
        html.Div([
            dcc.Graph(id="all-conversations-summary", figure=create_all_conversations_summary_chart(data)),
            html.P("Average Commitment Escalation, Belief Shifting, and Total Risk scores across all 30 conversations with peak turn marked",
                   style={"textAlign": "center", "fontSize": "12px", "color": "#666", "fontStyle": "italic", "marginTop": "10px"})
        ], style={"width": "90%", "margin": "auto", "padding": "10px"}),
    ]),

    html.Hr(),

    # Row 6: Individual Conversation Viewer
    html.H2("Individual Conversation Analysis", style={"marginTop": "30px"}),

    html.Div([
        html.Label("Select Conversation:", style={"fontWeight": "bold", "marginRight": "10px"}),
        dcc.Dropdown(
            id="scenario-dropdown",
            options=[{"label": sid, "value": sid} for sid in data.get_scenario_ids()],
            value=data.get_scenario_ids()[0] if data.get_scenario_ids() else None,
            style={"width": "400px"}
        )
    ], style={"marginBottom": "20px"}),

    # Timeline chart
    html.Div([
        dcc.Graph(id="timeline-chart")
    ]),

    html.Hr(),

    # Conversation detail
    html.H3("Turn-by-Turn Detail"),
    html.Div(id="conversation-detail", style={"marginTop": "20px"})

], style={"padding": "30px", "fontFamily": "Arial, sans-serif"})


# Callbacks
@app.callback(
    [Output("timeline-chart", "figure"),
     Output("conversation-detail", "children")],
    [Input("scenario-dropdown", "value")]
)
def update_conversation_view(scenario_id):
    """Update timeline and detail view when scenario changes."""
    if not scenario_id:
        return go.Figure(), html.Div("No data available")

    analysis = data.get_analysis(scenario_id)

    timeline_fig = create_timeline_chart(analysis)
    detail_view = create_conversation_detail(analysis)

    return timeline_fig, detail_view


if __name__ == "__main__":
    print("Starting Manipulation Tracker Dashboard...")
    print(f"Loaded {len(data.analyses)} conversations")
    print("Open browser to: http://127.0.0.1:8050")
    app.run(debug=True, port=8050)
