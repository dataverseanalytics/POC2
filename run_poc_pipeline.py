"""
End-to-End Pipeline for Membership Renewal Prediction
Trains model and generates all POC outputs
"""

import pandas as pd
import numpy as np
import os
from datetime import datetime
import json

from feature_engineering import prepare_training_data, prepare_scoring_data
from renewal_model import RenewalPredictor, evaluate_model
from poc_outputs import POCOutputGenerator


def convert_to_serializable(obj):
    """Convert numpy types to Python native types for JSON serialization"""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, pd.Series):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_to_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(item) for item in obj]
    return obj



def create_models_directory():
    """Create models directory if it doesn't exist"""
    if not os.path.exists('models'):
        os.makedirs('models')
        print("✅ Created 'models' directory")


def train_renewal_model(
    training_data_path: str,
    model_save_path: str = 'models/renewal_predictor.pkl'
):
    """
    Train the membership renewal prediction model
    
    Args:
        training_data_path: Path to labeled training data CSV
        model_save_path: Path to save trained model
        
    Returns:
        Trained predictor and feature engine
    """
    print("\n" + "=" * 70)
    print("STEP 1: TRAINING MEMBERSHIP RENEWAL PREDICTION MODEL")
    print("=" * 70)
    
    # Load training data
    print(f"\n📂 Loading training data from: {training_data_path}")
    df_train = pd.read_csv(training_data_path)
    print(f"   Loaded {len(df_train):,} records")
    print(f"   Target distribution: {df_train['label_renewed_within_60d'].value_counts().to_dict()}")
    
    # Prepare features
    print("\n🔧 Engineering features...")
    X, y, engine = prepare_training_data(df_train)
    print(f"   Created {X.shape[1]} features")
    
    # Get categorical features
    categorical_features = engine.get_feature_names()['categorical']
    print(f"   Categorical features: {len(categorical_features)}")
    print(f"   Numerical features: {len(engine.get_feature_names()['numerical'])}")
    
    # Train model
    print("\n🚀 Training LightGBM model...")
    predictor = RenewalPredictor(random_state=42)
    metrics = predictor.train(
        X, y,
        categorical_features=categorical_features,
        validation_split=0.2
    )
    
    # Save model
    create_models_directory()
    predictor.save_model(model_save_path)
    
    print(f"\n✅ Model training complete!")
    print(f"   Validation AUC: {metrics['val_auc']:.4f}")
    
    return predictor, engine, metrics


def generate_predictions(
    predictor: RenewalPredictor,
    engine,
    scoring_data_path: str,
    output_dir: str = 'outputs'
):
    """
    Generate predictions and all POC outputs
    
    Args:
        predictor: Trained model
        engine: Feature engineering engine
        scoring_data_path: Path to unlabeled scoring data
        output_dir: Directory to save outputs
        
    Returns:
        Dictionary of all outputs
    """
    print("\n" + "=" * 70)
    print("STEP 2: GENERATING PREDICTIONS AND POC OUTPUTS")
    print("=" * 70)
    
    # Load scoring data
    print(f"\n📂 Loading scoring data from: {scoring_data_path}")
    df_score = pd.read_csv(scoring_data_path)
    print(f"   Loaded {len(df_score):,} members to score")
    
    # Prepare features
    print("\n🔧 Engineering features for scoring data...")
    X_score, df_features = prepare_scoring_data(df_score, engine)
    print(f"   Feature matrix shape: {X_score.shape}")
    
    # Get categorical features
    categorical_features = engine.get_feature_names()['categorical']
    
    # Generate predictions
    print("\n🎯 Generating renewal probability predictions...")
    probabilities = predictor.predict_proba(
        X_score,
        categorical_features=categorical_features,
        use_calibrated=True
    )
    print(f"   Average predicted probability: {probabilities.mean():.2%}")
    
    # Calculate SHAP values (sample for performance)
    print("\n🔍 Calculating SHAP values for explainability...")
    sample_size = min(500, len(X_score))
    sample_indices = np.random.choice(len(X_score), sample_size, replace=False)
    X_sample = X_score.iloc[sample_indices]
    
    shap_values = predictor.get_shap_values(
        X_sample,
        categorical_features=categorical_features
    )
    print(f"   Calculated SHAP values for {sample_size} members")
    
    # Initialize POC output generator
    generator = POCOutputGenerator()
    
    # Generate all outputs
    outputs = {}
    
    # Output 1: Renewal Probability Scores
    print("\n📊 Generating Output 1: Renewal Probability Scores...")
    outputs['renewal_scores'] = generator.generate_renewal_scores(
        df_score['member_id'],
        probabilities
    )
    print(f"   ✅ Generated scores for {len(outputs['renewal_scores'])} members")
    
    # Output 2: Risk Segmentation
    print("\n📊 Generating Output 2: Risk Segmentation...")
    outputs['risk_segments'] = generator.create_risk_segments(
        probabilities,
        df_score
    )
    print(f"   ✅ Risk distribution:")
    for level, stats in outputs['risk_segments'].items():
        if level != 'overall':
            print(f"      {level}: {stats['count']} members ({stats['percentage']}%)")
    
    # Output 3: Key Drivers (for sample members)
    print("\n📊 Generating Output 3: Key Drivers (sample)...")
    outputs['driver_examples'] = []
    for i in range(min(10, len(sample_indices))):
        idx = sample_indices[i]
        drivers = generator.generate_driver_explanations(
            df_score.iloc[idx]['member_id'],
            shap_values[i],
            X_score.columns.tolist(),
            X_score.iloc[idx],
            top_n=5
        )
        outputs['driver_examples'].append({
            'member_id': df_score.iloc[idx]['member_id'],
            'renewal_probability': round(probabilities[idx] * 100, 1),
            'drivers': drivers
        })
    print(f"   ✅ Generated driver explanations for {len(outputs['driver_examples'])} sample members")
    
    # Output 4: Engagement Health Scores
    print("\n📊 Generating Output 4: Engagement Health Scores...")
    outputs['engagement_scores'] = generator.calculate_engagement_score(df_features)
    print(f"   ✅ Calculated engagement scores for {len(outputs['engagement_scores'])} members")
    print(f"      Average score: {outputs['engagement_scores']['engagement_health_score'].mean():.1f}")
    
    # Output 5: What-If Scenarios (example)
    print("\n📊 Generating Output 5: What-If Scenario Examples...")
    outputs['scenario_examples'] = []
    high_risk_members = outputs['renewal_scores'][
        outputs['renewal_scores']['risk_level'] == 'High Risk'
    ].head(5)
    
    for _, member_row in high_risk_members.iterrows():
        member_id = member_row['member_id']
        member_idx = df_score[df_score['member_id'] == member_id].index[0]
        
        scenario = generator.simulate_intervention(
            df_features.iloc[member_idx],
            probabilities[member_idx],
            {
                'attend_event': 2,
                'portal_login': 3,
                'enable_auto_renew': 1
            },
            predictor,
            X_score.columns.tolist()
        )
        scenario['member_id'] = member_id
        outputs['scenario_examples'].append(scenario)
    
    print(f"   ✅ Generated {len(outputs['scenario_examples'])} what-if scenario examples")
    
    # Output 6: Executive Summary
    print("\n📊 Generating Output 6: Executive Portfolio View...")
    outputs['executive_summary'] = generator.generate_executive_summary(
        outputs['renewal_scores'],
        df_features
    )
    print(f"   ✅ Executive summary generated")
    print(f"      Expected renewals: {outputs['executive_summary']['overview']['expected_renewals']}")
    print(f"      Expected renewal rate: {outputs['executive_summary']['overview']['expected_renewal_rate']}%")
    
    # Output 7: CRM Action Recommendations
    print("\n📊 Generating Output 7: CRM Action Recommendations...")
    outputs['crm_actions'] = generator.generate_crm_actions(
        outputs['renewal_scores'],
        df_features
    )
    print(f"   ✅ Generated CRM actions for {len(outputs['crm_actions'])} members")
    print(f"      High priority actions: {(outputs['crm_actions']['priority'] == 'High').sum()}")
    
    # Save outputs
    print(f"\n💾 Saving outputs to '{output_dir}' directory...")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Save CSV outputs
    outputs['renewal_scores'].to_csv(f'{output_dir}/renewal_scores.csv', index=False)
    outputs['engagement_scores'].to_csv(f'{output_dir}/engagement_scores.csv', index=False)
    outputs['crm_actions'].to_csv(f'{output_dir}/crm_actions.csv', index=False)
    
    # Save JSON outputs (convert numpy types first)
    with open(f'{output_dir}/risk_segments.json', 'w') as f:
        json.dump(convert_to_serializable(outputs['risk_segments']), f, indent=2)
    
    with open(f'{output_dir}/executive_summary.json', 'w') as f:
        json.dump(convert_to_serializable(outputs['executive_summary']), f, indent=2)
    
    with open(f'{output_dir}/driver_examples.json', 'w') as f:
        json.dump(convert_to_serializable(outputs['driver_examples']), f, indent=2)
    
    with open(f'{output_dir}/scenario_examples.json', 'w') as f:
        json.dump(convert_to_serializable(outputs['scenario_examples']), f, indent=2)
    
    print("   ✅ All outputs saved successfully!")
    
    return outputs


def main():
    """Main execution pipeline"""
    
    print("\n" + "=" * 70)
    print("MEMBERSHIP RENEWAL AI - POC PIPELINE")
    print("=" * 70)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # File paths
    training_data = 'membership_renewal_history_labeled_2022_2024_12000rows_v4.csv'
    scoring_data = 'membership_renewal_scoring_2025_unlabeled_4000rows_v4.csv'
    model_path = 'models/renewal_predictor.pkl'
    
    # Step 1: Train model
    predictor, engine, metrics = train_renewal_model(training_data, model_path)
    
    # Step 2: Generate predictions and outputs
    outputs = generate_predictions(predictor, engine, scoring_data)
    
    # Summary
    print("\n" + "=" * 70)
    print("✅ POC PIPELINE COMPLETE!")
    print("=" * 70)
    print(f"\n📊 Summary:")
    print(f"   Model AUC: {metrics['val_auc']:.4f}")
    print(f"   Members scored: {len(outputs['renewal_scores']):,}")
    print(f"   High risk members: {(outputs['renewal_scores']['risk_level'] == 'High Risk').sum():,}")
    print(f"   Expected renewal rate: {outputs['executive_summary']['overview']['expected_renewal_rate']}%")
    print(f"\n📁 Outputs saved to 'outputs/' directory")
    print(f"   - renewal_scores.csv")
    print(f"   - engagement_scores.csv")
    print(f"   - crm_actions.csv")
    print(f"   - risk_segments.json")
    print(f"   - executive_summary.json")
    print(f"   - driver_examples.json")
    print(f"   - scenario_examples.json")
    print(f"\nCompleted at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)


if __name__ == "__main__":
    main()
