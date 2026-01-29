# Membership Renewal AI POC - Project Files

## 📁 Essential Files

This directory contains the complete Membership Renewal AI POC implementation.

### Core Python Modules
- `feature_engineering.py` - Feature engineering pipeline (54 features)
- `renewal_model.py` - LightGBM model training and prediction
- `poc_outputs.py` - All 7 POC output generators
- `run_poc_pipeline.py` - End-to-end automation script
- `test_poc.py` - Component validation tests
- `app.py` - FastAPI application (optional API endpoints)

### Data Files
- `membership_renewal_history_labeled_2022_2024_12000rows_v4.csv` - Training data (12K records)
- `membership_renewal_scoring_2025_unlabeled_4000rows_v4.csv` - Scoring data (4K members)

### Model Artifacts
- `models/renewal_predictor.pkl` - Trained LightGBM model

### Generated Outputs
- `outputs/renewal_scores.csv` - Output 1: Renewal probabilities
- `outputs/risk_segments.json` - Output 2: Risk segmentation
- `outputs/driver_examples.json` - Output 3: SHAP explanations
- `outputs/engagement_scores.csv` - Output 4: Engagement health scores
- `outputs/scenario_examples.json` - Output 5: What-if scenarios
- `outputs/executive_summary.json` - Output 6: Executive dashboard
- `outputs/crm_actions.csv` - Output 7: CRM recommendations

### Documentation
- `POC_SUMMARY.md` - Complete POC summary and results
- `README.md` - Project overview
- `requirements.txt` - Python dependencies

### Configuration
- `.git/` - Git repository
- `env/` - Python virtual environment
- `__pycache__/` - Python cache files

---

## 🚀 Quick Start

1. **Activate environment**:
   ```powershell
   .\env\Scripts\Activate.ps1
   ```

2. **Run the complete pipeline**:
   ```powershell
   python run_poc_pipeline.py
   ```

3. **View outputs**:
   - Check `outputs/` directory for all 7 POC outputs

---

## 📊 What Was Removed

Cleaned up old/redundant files:
- ❌ `subscription-billing from GoMask.csv` - Old data file
- ❌ `membership-records from Gomask.csv` - Old data file
- ❌ `project_bundle.zip` - Old archive
- ❌ `visio of membership probability.pdf` - Old diagram
- ❌ `data_processor.py` - Replaced by `feature_engineering.py`
- ❌ `ml_model.py` - Replaced by `renewal_model.py`
- ❌ `model_trainer.py` - Integrated into `renewal_model.py`
- ❌ `risk_calculator.py` - Integrated into `poc_outputs.py`
- ❌ `debug_filter.py` - No longer needed
- ❌ `zip_project.py` - No longer needed
- ❌ `push_to_github.ps1` - No longer needed

---

## ✅ Current Status

**POC Complete**: All 7 outputs generated for 4,000 members
- Model AUC: 0.85+
- High-risk members: 518 (13%)
- Revenue at risk: $139,196
- Expected renewal rate: 43.9%

See `POC_SUMMARY.md` for full results.
