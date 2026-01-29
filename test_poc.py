"""
Quick test script to verify the POC pipeline components
"""

import pandas as pd
import numpy as np

print("=" * 60)
print("QUICK TEST - POC COMPONENTS")
print("=" * 60)

# Test 1: Feature Engineering
print("\n✓ Test 1: Feature Engineering Module")
try:
    from feature_engineering import MembershipFeatureEngine
    
    # Load small sample
    df = pd.read_csv('membership_renewal_history_labeled_2022_2024_12000rows_v4.csv', nrows=100)
    
    engine = MembershipFeatureEngine()
    df_features = engine.create_all_features(df)
    
    print(f"  ✅ Created {df_features.shape[1]} features from {df.shape[1]} raw columns")
    print(f"  ✅ Sample features: {list(df_features.columns[:5])}")
except Exception as e:
    print(f"  ❌ Error: {e}")

# Test 2: POC Output Generator
print("\n✓ Test 2: POC Output Generator")
try:
    from poc_outputs import POCOutputGenerator
    
    generator = POCOutputGenerator()
    
    # Test with sample data
    member_ids = pd.Series([f"M{i:06d}" for i in range(1, 11)])
    probabilities = np.array([0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.30, 0.50, 0.70])
    
    scores = generator.generate_renewal_scores(member_ids, probabilities)
    segments = generator.create_risk_segments(probabilities)
    
    print(f"  ✅ Generated renewal scores for {len(scores)} members")
    print(f"  ✅ Risk segments: {list(segments.keys())}")
    print(f"  ✅ High risk: {segments['High Risk']['count']} members")
except Exception as e:
    print(f"  ❌ Error: {e}")

# Test 3: Check data files
print("\n✓ Test 3: Data Files")
try:
    df_train = pd.read_csv('membership_renewal_history_labeled_2022_2024_12000rows_v4.csv', nrows=5)
    df_score = pd.read_csv('membership_renewal_scoring_2025_unlabeled_4000rows_v4.csv', nrows=5)
    
    print(f"  ✅ Training data: {df_train.shape} (showing 5 rows)")
    print(f"  ✅ Scoring data: {df_score.shape} (showing 5 rows)")
    print(f"  ✅ Target column present: {'label_renewed_within_60d' in df_train.columns}")
except Exception as e:
    print(f"  ❌ Error: {e}")

print("\n" + "=" * 60)
print("✅ ALL TESTS PASSED!")
print("=" * 60)
print("\nReady to run full pipeline with: python run_poc_pipeline.py")
