# 🎯 Running the Interactive Dashboard

## Quick Start

1. **Make sure outputs are generated**:
   ```powershell
   python run_poc_pipeline.py
   ```

2. **Start the dashboard**:
   ```powershell
   python dashboard.py
   ```

3. **Open your browser**:
   - Navigate to: `http://127.0.0.1:8050`

---

## Dashboard Features

### 📊 Tab 1: Member Search
- Search for any member by ID (e.g., M000001)
- View renewal probability, risk level, and engagement score
- See top drivers (SHAP explanations)
- Get recommended CRM actions

### 📈 Tab 2: Risk Analysis
- **Pie Chart**: Visual distribution of High/Medium/Low risk members
- **Histogram**: Probability distribution across all members
- **Statistics Table**: Detailed stats for each risk segment

### 🔮 Tab 3: What-If Scenarios
- Select a member ID
- Choose interventions:
  - Attend 2 Events
  - Join Committee
  - 3 Portal Logins
  - Enable Auto-Renew
  - Attend Webinar
- See projected probability change
- Get recommendation on intervention effectiveness

### 📞 Tab 4: CRM Actions
- View all recommended actions
- Filter by priority (High/Medium/Low)
- See action type, channel, and timeline
- Export-ready for CRM import

### 💼 Tab 5: Executive Summary
- Portfolio overview metrics
- Financial impact analysis
- Risk distribution breakdown
- C-suite dashboard view

---

## Dashboard Screenshots

### KPI Cards (Top of Dashboard)
- Total Members: 4,000
- Expected Renewals: 1,757 (43.9%)
- High Risk Members: 518 (13%)
- Revenue at Risk: $139,196

### Interactive Features
- Real-time member search
- Dynamic charts and graphs
- What-if scenario simulator
- Filterable CRM actions table

---

## Troubleshooting

### "Data not loaded" error
**Solution**: Run the pipeline first
```powershell
python run_poc_pipeline.py
```

### Port already in use
**Solution**: Change the port in dashboard.py (line 485):
```python
app.run_server(debug=True, host='127.0.0.1', port=8051)  # Changed from 8050
```

### Missing dependencies
**Solution**: Install requirements
```powershell
pip install -r requirements.txt
```

---

## Tips for Testing

1. **Try different members**:
   - High risk: M000095, M000276, M000380
   - Medium risk: M000001, M000002, M000003
   - Search any member from the outputs

2. **Test what-if scenarios**:
   - Start with a high-risk member
   - Add multiple interventions
   - See the cumulative impact

3. **Explore risk analysis**:
   - Check the pie chart for overall distribution
   - Use histogram to see probability clusters
   - Review statistics table for segment details

4. **Filter CRM actions**:
   - Start with "High Priority Only"
   - Review recommended actions
   - Check timeline for urgency

---

## Stopping the Dashboard

Press `Ctrl+C` in the terminal to stop the server.

---

**Dashboard Built With**: Plotly Dash + Bootstrap  
**Last Updated**: January 2026
