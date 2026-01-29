"""
Feature Engineering Pipeline for Membership Renewal Prediction
Creates comprehensive features from raw membership data
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')


class MembershipFeatureEngine:
    """
    Comprehensive feature engineering for membership renewal prediction
    """
    
    def __init__(self):
        self.feature_names = []
        self.categorical_features = []
        self.numerical_features = []
        
    def create_all_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create all engineered features
        
        Args:
            df: Raw membership dataframe
            
        Returns:
            DataFrame with all engineered features
        """
        df = df.copy()
        
        # Create feature groups
        df = self._create_engagement_features(df)
        df = self._create_behavioral_features(df)
        df = self._create_lifecycle_features(df)
        df = self._create_risk_features(df)
        df = self._create_financial_features(df)
        df = self._create_demographic_features(df)
        
        # Store feature lists
        self._identify_feature_types(df)
        
        return df
    
    def _create_engagement_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create engagement-related features"""
        
        # Total engagement score (weighted sum)
        df['total_engagement_score'] = (
            df['events_12m'] * 2.0 +
            df['webinars_12m'] * 1.5 +
            df['courses_12m'] * 2.5 +
            df['website_sessions_90d'] * 0.5 +
            df['community_logins_90d'] * 0.8
        )
        
        # Engagement intensity (normalized)
        df['engagement_intensity'] = df['total_engagement_score'] / (df['tenure_years'] + 1)
        
        # Email engagement score
        df['email_engagement'] = (df['email_open_rate'] + df['email_click_rate'] * 2) / 3
        
        # Activity flags
        df['is_active_event_attendee'] = (df['events_12m'] >= 2).astype(int)
        df['is_webinar_participant'] = (df['webinars_12m'] >= 1).astype(int)
        df['is_course_taker'] = (df['courses_12m'] >= 1).astype(int)
        df['is_portal_user'] = (df['website_sessions_90d'] >= 3).astype(int)
        
        # Engagement diversity (how many channels they use)
        df['engagement_channels'] = (
            (df['events_12m'] > 0).astype(int) +
            (df['webinars_12m'] > 0).astype(int) +
            (df['courses_12m'] > 0).astype(int) +
            (df['website_sessions_90d'] > 0).astype(int) +
            (df['community_logins_90d'] > 0).astype(int)
        )
        
        # Zero engagement flag (high risk)
        df['zero_engagement'] = (df['total_engagement_score'] == 0).astype(int)
        
        return df
    
    def _create_behavioral_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create behavioral pattern features"""
        
        # Committee participation score
        df['committee_score'] = (
            df['committee_member_flag'] * 10 +
            df['committee_leadership_flag'] * 20 +
            df['committee_meetings_12m'] * 2
        )
        
        # Leadership bonus
        df['has_leadership_role'] = df['committee_leadership_flag']
        
        # Donation behavior
        df['is_donor'] = (df['donation_amount_12m'] > 0).astype(int)
        df['donation_tier'] = pd.cut(
            df['donation_amount_12m'],
            bins=[-0.1, 0, 50, 150, 500, np.inf],
            labels=['none', 'small', 'medium', 'large', 'major']
        )
        
        # Auto-renew advantage
        df['auto_renew_enabled'] = df['auto_renew_flag']
        
        # Support interaction intensity
        df['support_intensity'] = df['support_tickets_12m'] + df['complaints_12m'] * 3
        
        # Has complaints flag (negative signal)
        df['has_complaints'] = (df['complaints_12m'] > 0).astype(int)
        
        return df
    
    def _create_lifecycle_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create membership lifecycle features"""
        
        # Tenure buckets
        df['tenure_bucket'] = pd.cut(
            df['tenure_years'],
            bins=[-1, 2, 5, 10, 20, np.inf],
            labels=['new', 'established', 'veteran', 'long_term', 'legacy']
        )
        
        # Lifecycle stage
        df['lifecycle_stage'] = 'mature'
        df.loc[df['tenure_years'] <= 1, 'lifecycle_stage'] = 'onboarding'
        df.loc[(df['tenure_years'] > 1) & (df['tenure_years'] <= 3), 'lifecycle_stage'] = 'growth'
        df.loc[df['tenure_years'] > 10, 'lifecycle_stage'] = 'loyal'
        
        # Renewal history
        df['renewed_last_year'] = df['renewed_prev_year']
        
        # Member age groups
        df['age_group'] = pd.cut(
            df['member_age'],
            bins=[0, 30, 45, 60, 100],
            labels=['young', 'mid_career', 'senior', 'retired']
        )
        
        # New member flag (first 2 years)
        df['is_new_member'] = (df['tenure_years'] <= 2).astype(int)
        
        return df
    
    def _create_risk_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create risk indicator features"""
        
        # Payment risk score
        df['payment_risk_score'] = (
            df['failed_payment_flag'] * 50 +
            (df['invoice_age_days'] > 60).astype(int) * 30 +
            (df['invoice_age_days'] > 90).astype(int) * 20
        )
        
        # High invoice age flag
        df['overdue_invoice'] = (df['invoice_age_days'] > 60).astype(int)
        
        # Negative interaction score
        df['negative_signals'] = (
            df['failed_payment_flag'] +
            df['has_complaints'] +
            df['zero_engagement'] +
            (1 - df['renewed_prev_year'])
        )
        
        # At-risk flag (multiple negative signals)
        df['high_risk_flag'] = (df['negative_signals'] >= 2).astype(int)
        
        return df
    
    def _create_financial_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create financial/pricing features"""
        
        # Dues tier
        df['dues_tier'] = pd.cut(
            df['dues_amount'],
            bins=[0, 100, 300, 600, np.inf],
            labels=['budget', 'standard', 'premium', 'corporate']
        )
        
        # Price per year of membership
        df['price_per_tenure_year'] = df['dues_amount'] / (df['tenure_years'] + 1)
        
        # High value member (premium + engaged)
        df['is_high_value'] = (
            (df['dues_amount'] > 300) & 
            (df['total_engagement_score'] > 10)
        ).astype(int)
        
        return df
    
    def _create_demographic_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create demographic features"""
        
        # International flag
        df['is_international'] = (~df['country'].isin(['USA', 'Canada'])).astype(int)
        
        # Membership type encoding (already categorical)
        # Keep as-is for now, will be encoded later
        
        return df
    
    def _identify_feature_types(self, df: pd.DataFrame):
        """Identify categorical and numerical features"""
        
        # Categorical features
        self.categorical_features = [
            'country', 'chapter', 'membership_type',
            'committee_leadership_role', 'donation_tier',
            'tenure_bucket', 'lifecycle_stage', 'age_group', 'dues_tier'
        ]
        
        # Numerical features (excluding target and ID)
        exclude_cols = ['member_id', 'cycle_year', 'renewal_due_date', 
                       'renewal_date', 'label_renewed_within_60d']
        
        self.numerical_features = [
            col for col in df.columns 
            if col not in self.categorical_features + exclude_cols
            and df[col].dtype in ['int64', 'float64']
        ]
    
    def get_feature_names(self) -> Dict[str, List[str]]:
        """Get lists of feature names by type"""
        return {
            'categorical': self.categorical_features,
            'numerical': self.numerical_features,
            'all': self.categorical_features + self.numerical_features
        }


def prepare_training_data(
    df: pd.DataFrame,
    target_col: str = 'label_renewed_within_60d'
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Prepare data for model training
    
    Args:
        df: Feature-engineered dataframe
        target_col: Name of target column
        
    Returns:
        Tuple of (features_df, target_series)
    """
    # Create feature engine
    engine = MembershipFeatureEngine()
    df_features = engine.create_all_features(df)
    
    # Get feature names
    feature_info = engine.get_feature_names()
    feature_cols = feature_info['all']
    
    # Separate features and target
    X = df_features[feature_cols].copy()
    y = df_features[target_col].copy()
    
    # Handle missing values in categorical features
    for col in feature_info['categorical']:
        if col in X.columns:
            X[col] = X[col].fillna('unknown')
    
    # Handle missing values in numerical features
    for col in feature_info['numerical']:
        if col in X.columns:
            X[col] = X[col].fillna(X[col].median())
    
    return X, y, engine


def prepare_scoring_data(
    df: pd.DataFrame,
    engine: MembershipFeatureEngine
) -> pd.DataFrame:
    """
    Prepare unlabeled data for scoring
    
    Args:
        df: Raw scoring dataframe
        engine: Fitted feature engine from training
        
    Returns:
        Feature-engineered dataframe ready for prediction
    """
    # Create features
    df_features = engine.create_all_features(df)
    
    # Get feature names
    feature_info = engine.get_feature_names()
    feature_cols = feature_info['all']
    
    # Select features
    X = df_features[feature_cols].copy()
    
    # Handle missing values
    for col in feature_info['categorical']:
        if col in X.columns:
            X[col] = X[col].fillna('unknown')
    
    for col in feature_info['numerical']:
        if col in X.columns:
            X[col] = X[col].fillna(X[col].median())
    
    return X, df_features


if __name__ == "__main__":
    # Test feature engineering
    print("Loading training data...")
    df_train = pd.read_csv('membership_renewal_history_labeled_2022_2024_12000rows_v4.csv')
    
    print(f"Training data shape: {df_train.shape}")
    print(f"Target distribution:\n{df_train['label_renewed_within_60d'].value_counts()}")
    
    print("\nCreating features...")
    X, y, engine = prepare_training_data(df_train)
    
    print(f"\nFeature matrix shape: {X.shape}")
    print(f"Number of features: {len(engine.get_feature_names()['all'])}")
    print(f"Categorical features: {len(engine.get_feature_names()['categorical'])}")
    print(f"Numerical features: {len(engine.get_feature_names()['numerical'])}")
    
    print("\n✅ Feature engineering complete!")
