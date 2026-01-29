"""
Interactive Dashboard for Membership Renewal AI
Built with Plotly Dash for testing and visualization
"""

import dash
from dash import dcc, html, Input, Output, State, dash_table
import dash_bootstrap_components as dbc
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import json
from datetime import datetime

# Initialize Dash app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
app.title = "Membership Renewal AI Dashboard"

# Load data
try:
    df_scores = pd.read_csv('outputs/renewal_scores.csv')
    df_engagement = pd.read_csv('outputs/engagement_scores.csv')
    df_crm = pd.read_csv('outputs/crm_actions.csv')
    
    with open('outputs/executive_summary.json', 'r') as f:
        exec_summary = json.load(f)
    
    with open('outputs/risk_segments.json', 'r') as f:
        risk_segments = json.load(f)
    
    with open('outputs/driver_examples.json', 'r') as f:
        driver_examples = json.load(f)
        
    data_loaded = True
except Exception as e:
    data_loaded = False
    error_message = str(e)

# Dashboard Layout
app.layout = dbc.Container([
    dbc.Row([
        dbc.Col([
            html.H1("🎯 Membership Renewal AI Dashboard", className="text-center mb-4 mt-4"),
            html.P("Interactive testing and visualization of renewal predictions", 
                   className="text-center text-muted mb-4")
        ])
    ]),
    
    # KPI Cards
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H4("Total Members", className="card-title"),
                    html.H2(f"{exec_summary['overview']['total_members']:,}" if data_loaded else "N/A", 
                           className="text-primary"),
                ])
            ])
        ], width=3),
        
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H4("Expected Renewals", className="card-title"),
                    html.H2(f"{exec_summary['overview']['expected_renewals']:,}" if data_loaded else "N/A", 
                           className="text-success"),
                    html.P(f"{exec_summary['overview']['expected_renewal_rate']}%" if data_loaded else "", 
                          className="text-muted")
                ])
            ])
        ], width=3),
        
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H4("High Risk Members", className="card-title"),
                    html.H2(f"{exec_summary['risk_distribution']['high_risk']['count']:,}" if data_loaded else "N/A", 
                           className="text-danger"),
                    html.P(f"{exec_summary['risk_distribution']['high_risk']['percentage']}%" if data_loaded else "", 
                          className="text-muted")
                ])
            ])
        ], width=3),
        
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H4("Revenue at Risk", className="card-title"),
                    html.H2(f"${exec_summary['financial_impact']['revenue_at_risk']:,.0f}" if data_loaded else "N/A", 
                           className="text-warning"),
                ])
            ])
        ], width=3),
    ], className="mb-4"),
    
    # Tabs for different views
    dbc.Tabs([
        # Tab 1: Member Search
        dbc.Tab([
            dbc.Row([
                dbc.Col([
                    html.H4("🔍 Member Search & Details", className="mt-4 mb-3"),
                    dbc.InputGroup([
                        dbc.Input(id="member-search", placeholder="Enter Member ID (e.g., M000001)", type="text"),
                        dbc.Button("Search", id="search-button", color="primary"),
                    ], className="mb-3"),
                    
                    html.Div(id="member-details"),
                ], width=12)
            ])
        ], label="Member Search"),
        
        # Tab 2: Risk Distribution
        dbc.Tab([
            dbc.Row([
                dbc.Col([
                    html.H4("📊 Risk Distribution Analysis", className="mt-4 mb-3"),
                    dcc.Graph(id="risk-pie-chart"),
                ], width=6),
                
                dbc.Col([
                    html.H4("📈 Probability Distribution", className="mt-4 mb-3"),
                    dcc.Graph(id="probability-histogram"),
                ], width=6),
            ]),
            
            dbc.Row([
                dbc.Col([
                    html.H4("📋 Risk Segment Statistics", className="mt-4 mb-3"),
                    html.Div(id="risk-stats-table"),
                ], width=12)
            ])
        ], label="Risk Analysis"),
        
        # Tab 3: What-If Scenarios
        dbc.Tab([
            dbc.Row([
                dbc.Col([
                    html.H4("🔮 What-If Scenario Simulator", className="mt-4 mb-3"),
                    
                    dbc.InputGroup([
                        dbc.InputGroupText("Member ID:"),
                        dbc.Input(id="scenario-member-id", placeholder="M000001", type="text"),
                    ], className="mb-3"),
                    
                    html.H5("Select Interventions:", className="mt-3"),
                    
                    dbc.Checklist(
                        id="intervention-checklist",
                        options=[
                            {"label": "Attend 2 Events", "value": "events"},
                            {"label": "Join Committee", "value": "committee"},
                            {"label": "3 Portal Logins", "value": "portal"},
                            {"label": "Enable Auto-Renew", "value": "autorenew"},
                            {"label": "Attend Webinar", "value": "webinar"},
                        ],
                        value=[],
                        className="mb-3"
                    ),
                    
                    dbc.Button("Run Scenario", id="run-scenario-button", color="success", className="mb-3"),
                    
                    html.Div(id="scenario-results"),
                ], width=12)
            ])
        ], label="What-If Scenarios"),
        
        # Tab 4: CRM Actions
        dbc.Tab([
            dbc.Row([
                dbc.Col([
                    html.H4("📞 CRM Action Recommendations", className="mt-4 mb-3"),
                    
                    dbc.RadioItems(
                        id="priority-filter",
                        options=[
                            {"label": "All Priorities", "value": "all"},
                            {"label": "High Priority Only", "value": "High"},
                            {"label": "Medium Priority Only", "value": "Medium"},
                            {"label": "Low Priority Only", "value": "Low"},
                        ],
                        value="all",
                        inline=True,
                        className="mb-3"
                    ),
                    
                    html.Div(id="crm-actions-table"),
                ], width=12)
            ])
        ], label="CRM Actions"),
        
        # Tab 5: Executive Summary
        dbc.Tab([
            dbc.Row([
                dbc.Col([
                    html.H4("📊 Executive Portfolio View", className="mt-4 mb-3"),
                    html.Div(id="executive-summary-content"),
                ], width=12)
            ])
        ], label="Executive Summary"),
    ]),
    
    html.Hr(),
    html.P(f"Dashboard loaded at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", 
           className="text-center text-muted mt-4 mb-4")
    
], fluid=True)


# Callbacks

# Member Search
@app.callback(
    Output("member-details", "children"),
    Input("search-button", "n_clicks"),
    State("member-search", "value"),
    prevent_initial_call=True
)
def search_member(n_clicks, member_id):
    if not member_id or not data_loaded:
        return dbc.Alert("Please enter a valid Member ID", color="warning")
    
    # Find member in scores
    member_score = df_scores[df_scores['member_id'] == member_id]
    
    if member_score.empty:
        return dbc.Alert(f"Member {member_id} not found", color="danger")
    
    # Get engagement score
    member_engagement = df_engagement[df_engagement['member_id'] == member_id]
    
    # Get drivers (if available in examples)
    drivers_info = None
    for example in driver_examples:
        if example['member_id'] == member_id:
            drivers_info = example['drivers']
            break
    
    # Build member card
    prob = member_score.iloc[0]['renewal_probability']
    risk = member_score.iloc[0]['risk_level']
    
    risk_color = "danger" if "High" in risk else "warning" if "Medium" in risk else "success"
    
    card = dbc.Card([
        dbc.CardHeader(html.H4(f"Member: {member_id}")),
        dbc.CardBody([
            dbc.Row([
                dbc.Col([
                    html.H5("Renewal Probability"),
                    html.H2(f"{prob}%", className=f"text-{risk_color}"),
                    dbc.Badge(risk, color=risk_color, className="mb-3"),
                ], width=4),
                
                dbc.Col([
                    html.H5("Engagement Health Score"),
                    html.H2(f"{member_engagement.iloc[0]['engagement_health_score']:.1f}" if not member_engagement.empty else "N/A"),
                    html.P(f"Percentile: {member_engagement.iloc[0]['percentile_rank']}" if not member_engagement.empty else ""),
                ], width=4),
                
                dbc.Col([
                    html.H5("Recommended Action"),
                    html.P(df_crm[df_crm['member_id'] == member_id].iloc[0]['action_type'] if not df_crm[df_crm['member_id'] == member_id].empty else "N/A"),
                    html.P(df_crm[df_crm['member_id'] == member_id].iloc[0]['timeline'] if not df_crm[df_crm['member_id'] == member_id].empty else "", className="text-muted"),
                ], width=4),
            ]),
            
            html.Hr(),
            
            html.H5("Top Drivers" if drivers_info else "Drivers not available in sample"),
            html.Ul([
                html.Li(driver['explanation']) for driver in (drivers_info[:5] if drivers_info else [])
            ]) if drivers_info else html.P("Run full pipeline to generate driver explanations for all members", className="text-muted"),
        ])
    ])
    
    return card


# Risk Pie Chart
@app.callback(
    Output("risk-pie-chart", "figure"),
    Input("risk-pie-chart", "id")
)
def update_risk_pie(id):
    if not data_loaded:
        return go.Figure()
    
    risk_counts = df_scores['risk_level'].value_counts()
    
    fig = px.pie(
        values=risk_counts.values,
        names=risk_counts.index,
        title="Member Distribution by Risk Level",
        color_discrete_map={
            'High Risk': '#dc3545',
            'Medium Risk': '#ffc107',
            'Low Risk': '#28a745'
        }
    )
    
    return fig


# Probability Histogram
@app.callback(
    Output("probability-histogram", "figure"),
    Input("probability-histogram", "id")
)
def update_histogram(id):
    if not data_loaded:
        return go.Figure()
    
    fig = px.histogram(
        df_scores,
        x='renewal_probability',
        nbins=50,
        title="Distribution of Renewal Probabilities",
        labels={'renewal_probability': 'Renewal Probability (%)'},
        color_discrete_sequence=['#007bff']
    )
    
    fig.add_vline(x=40, line_dash="dash", line_color="red", annotation_text="High Risk Threshold")
    fig.add_vline(x=70, line_dash="dash", line_color="green", annotation_text="Low Risk Threshold")
    
    return fig


# Risk Stats Table
@app.callback(
    Output("risk-stats-table", "children"),
    Input("risk-stats-table", "id")
)
def update_risk_stats(id):
    if not data_loaded:
        return html.P("Data not loaded")
    
    stats_data = []
    for level, stats in risk_segments.items():
        if level != 'overall':
            stats_data.append({
                'Risk Level': level,
                'Count': stats['count'],
                'Percentage': f"{stats['percentage']}%",
                'Avg Probability': f"{stats['avg_renewal_probability']}%",
                'Min Probability': f"{stats['min_probability']}%",
                'Max Probability': f"{stats['max_probability']}%",
            })
    
    return dash_table.DataTable(
        data=stats_data,
        columns=[{"name": i, "id": i} for i in stats_data[0].keys()],
        style_cell={'textAlign': 'left'},
        style_header={'backgroundColor': '#007bff', 'color': 'white', 'fontWeight': 'bold'},
        style_data_conditional=[
            {
                'if': {'filter_query': '{Risk Level} = "High Risk"'},
                'backgroundColor': '#f8d7da',
            },
            {
                'if': {'filter_query': '{Risk Level} = "Medium Risk"'},
                'backgroundColor': '#fff3cd',
            },
        ]
    )


# What-If Scenario
@app.callback(
    Output("scenario-results", "children"),
    Input("run-scenario-button", "n_clicks"),
    State("scenario-member-id", "value"),
    State("intervention-checklist", "value"),
    prevent_initial_call=True
)
def run_scenario(n_clicks, member_id, interventions):
    if not member_id or not data_loaded:
        return dbc.Alert("Please enter a Member ID", color="warning")
    
    # Get current probability
    member_score = df_scores[df_scores['member_id'] == member_id]
    if member_score.empty:
        return dbc.Alert(f"Member {member_id} not found", color="danger")
    
    current_prob = member_score.iloc[0]['renewal_probability']
    
    # Estimate impact (simplified)
    impact = 0
    intervention_list = []
    
    if 'events' in interventions:
        impact += 8
        intervention_list.append("Attend 2 events (+8%)")
    if 'committee' in interventions:
        impact += 12
        intervention_list.append("Join committee (+12%)")
    if 'portal' in interventions:
        impact += 5
        intervention_list.append("3 portal logins (+5%)")
    if 'autorenew' in interventions:
        impact += 15
        intervention_list.append("Enable auto-renew (+15%)")
    if 'webinar' in interventions:
        impact += 6
        intervention_list.append("Attend webinar (+6%)")
    
    projected_prob = min(current_prob + impact, 95)
    
    return dbc.Card([
        dbc.CardHeader(html.H5("Scenario Results")),
        dbc.CardBody([
            dbc.Row([
                dbc.Col([
                    html.H6("Current Probability"),
                    html.H3(f"{current_prob}%", className="text-warning"),
                ], width=4),
                
                dbc.Col([
                    html.H6("Projected Probability"),
                    html.H3(f"{projected_prob}%", className="text-success"),
                ], width=4),
                
                dbc.Col([
                    html.H6("Change"),
                    html.H3(f"+{impact}%", className="text-primary"),
                ], width=4),
            ]),
            
            html.Hr(),
            
            html.H6("Interventions Applied:"),
            html.Ul([html.Li(item) for item in intervention_list]) if intervention_list else html.P("No interventions selected", className="text-muted"),
            
            html.Hr(),
            
            dbc.Alert(
                "High impact intervention - Strongly recommended" if impact >= 15 else
                "Moderate impact - Recommended" if impact >= 8 else
                "Low impact - Consider other interventions",
                color="success" if impact >= 15 else "info" if impact >= 8 else "warning"
            )
        ])
    ])


# CRM Actions Table
@app.callback(
    Output("crm-actions-table", "children"),
    Input("priority-filter", "value")
)
def update_crm_table(priority):
    if not data_loaded:
        return html.P("Data not loaded")
    
    filtered_df = df_crm if priority == "all" else df_crm[df_crm['priority'] == priority]
    
    return dash_table.DataTable(
        data=filtered_df.head(50).to_dict('records'),
        columns=[
            {"name": "Member ID", "id": "member_id"},
            {"name": "Probability", "id": "renewal_probability"},
            {"name": "Risk Level", "id": "risk_level"},
            {"name": "Action Type", "id": "action_type"},
            {"name": "Priority", "id": "priority"},
            {"name": "Channel", "id": "channel"},
            {"name": "Timeline", "id": "timeline"},
        ],
        style_cell={'textAlign': 'left'},
        style_header={'backgroundColor': '#007bff', 'color': 'white', 'fontWeight': 'bold'},
        page_size=20,
        style_data_conditional=[
            {
                'if': {'filter_query': '{priority} = "High"'},
                'backgroundColor': '#f8d7da',
            },
        ]
    )


# Executive Summary
@app.callback(
    Output("executive-summary-content", "children"),
    Input("executive-summary-content", "id")
)
def update_executive_summary(id):
    if not data_loaded:
        return html.P("Data not loaded")
    
    return dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader(html.H5("Portfolio Overview")),
                dbc.CardBody([
                    html.P(f"Total Members: {exec_summary['overview']['total_members']:,}"),
                    html.P(f"Expected Renewals: {exec_summary['overview']['expected_renewals']:,}"),
                    html.P(f"Expected Renewal Rate: {exec_summary['overview']['expected_renewal_rate']}%"),
                    html.P(f"Average Probability: {exec_summary['overview']['avg_renewal_probability']}%"),
                ])
            ], className="mb-3"),
            
            dbc.Card([
                dbc.CardHeader(html.H5("Financial Impact")),
                dbc.CardBody([
                    html.P(f"Revenue at Risk: ${exec_summary['financial_impact']['revenue_at_risk']:,.2f}"),
                    html.P(f"High Risk Members: {exec_summary['financial_impact']['high_risk_members']:,}"),
                ])
            ])
        ], width=6),
        
        dbc.Col([
            dbc.Card([
                dbc.CardHeader(html.H5("Risk Distribution")),
                dbc.CardBody([
                    html.H6("High Risk", className="text-danger"),
                    html.P(f"{exec_summary['risk_distribution']['high_risk']['count']:,} members ({exec_summary['risk_distribution']['high_risk']['percentage']}%)"),
                    
                    html.H6("Medium Risk", className="text-warning mt-3"),
                    html.P(f"{exec_summary['risk_distribution']['medium_risk']['count']:,} members ({exec_summary['risk_distribution']['medium_risk']['percentage']}%)"),
                    
                    html.H6("Low Risk", className="text-success mt-3"),
                    html.P(f"{exec_summary['risk_distribution']['low_risk']['count']:,} members ({exec_summary['risk_distribution']['low_risk']['percentage']}%)"),
                ])
            ])
        ], width=6),
    ])


if __name__ == '__main__':
    if data_loaded:
        print("\n" + "="*60)
        print("🎯 MEMBERSHIP RENEWAL AI DASHBOARD")
        print("="*60)
        print(f"\n✅ Data loaded successfully!")
        print(f"   - {len(df_scores):,} members scored")
        print(f"   - {len(df_engagement):,} engagement scores")
        print(f"   - {len(df_crm):,} CRM actions")
        print(f"\n🌐 Starting dashboard server...")
        print(f"   Open your browser to: http://127.0.0.1:8050")
        print(f"\n" + "="*60 + "\n")
        
        app.run(debug=True, host='127.0.0.1', port=8050)
    else:
        print("\n❌ Error loading data:")
        print(f"   {error_message}")
        print("\n💡 Please run 'python run_poc_pipeline.py' first to generate outputs")

