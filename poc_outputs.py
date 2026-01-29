"""
POC Output Generators for Membership Renewal AI
Generates all 7 business-ready outputs
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')


class POCOutputGenerator:
    """
    Generate all 7 POC outputs for Membership Renewal AI
    """
    
    def __init__(self):
        self.risk_thresholds = {
            'high': 0.40,
            'medium': 0.70
        }
    
    # ========== OUTPUT 1: Renewal Probability Score ==========
    
    def generate_renewal_scores(
        self,
        member_ids: pd.Series,
        probabilities: np.ndarray
    ) -> pd.DataFrame:
        """
        Output 1: Renewal Probability Score
        
        Returns DataFrame with:
        - member_id
        - renewal_probability (%)
        - risk_level (High/Medium/Low)
        """
        # Convert probabilities to percentage
        prob_pct = probabilities * 100
        
        # Assign risk levels
        risk_levels = pd.cut(
            probabilities,
            bins=[0, self.risk_thresholds['high'], self.risk_thresholds['medium'], 1.0],
            labels=['High Risk', 'Medium Risk', 'Low Risk'],
            include_lowest=True
        )
        
        output = pd.DataFrame({
            'member_id': member_ids,
            'renewal_probability': prob_pct.round(1),
            'risk_level': risk_levels
        })
        
        return output
    
    # ========== OUTPUT 2: Risk Segmentation ==========
    
    def create_risk_segments(
        self,
        probabilities: np.ndarray,
        member_data: pd.DataFrame = None
    ) -> Dict:
        """
        Output 2: Risk Segmentation Buckets
        
        Returns statistics for each risk segment
        """
        # Assign risk levels
        risk_levels = pd.cut(
            probabilities,
            bins=[0, self.risk_thresholds['high'], self.risk_thresholds['medium'], 1.0],
            labels=['High Risk', 'Medium Risk', 'Low Risk'],
            include_lowest=True
        )
        
        # Calculate segment statistics
        segments = {}
        
        for level in ['High Risk', 'Medium Risk', 'Low Risk']:
            mask = risk_levels == level
            count = mask.sum()
            pct = (count / len(probabilities)) * 100
            avg_prob = probabilities[mask].mean() * 100 if count > 0 else 0
            
            segments[level] = {
                'count': int(count),
                'percentage': round(pct, 1),
                'avg_renewal_probability': round(avg_prob, 1),
                'min_probability': round(probabilities[mask].min() * 100, 1) if count > 0 else 0,
                'max_probability': round(probabilities[mask].max() * 100, 1) if count > 0 else 0
            }
        
        # Add overall statistics
        segments['overall'] = {
            'total_members': len(probabilities),
            'avg_probability': round(probabilities.mean() * 100, 1),
            'expected_renewals': int((probabilities >= 0.5).sum())
        }
        
        return segments
    
    # ========== OUTPUT 3: Key Drivers / Explainability ==========
    
    def generate_driver_explanations(
        self,
        member_id: str,
        shap_values: np.ndarray,
        feature_names: List[str],
        feature_values: pd.Series,
        top_n: int = 5
    ) -> List[Dict]:
        """
        Output 3: Key Drivers with Explainability
        
        Returns top N drivers for a member's renewal probability
        """
        # Get absolute SHAP values for ranking
        abs_shap = np.abs(shap_values)
        top_indices = np.argsort(abs_shap)[-top_n:][::-1]
        
        drivers = []
        
        for idx in top_indices:
            feature = feature_names[idx]
            shap_val = shap_values[idx]
            feature_val = feature_values.iloc[idx]
            
            # Calculate impact on probability (approximate)
            impact_pct = shap_val * 100
            direction = "positive" if shap_val > 0 else "negative"
            
            # Generate human-readable explanation
            explanation = self._format_driver_explanation(
                feature, feature_val, direction, abs(impact_pct)
            )
            
            drivers.append({
                'feature': feature,
                'feature_value': feature_val,
                'impact_direction': direction,
                'impact_magnitude': round(abs(impact_pct), 1),
                'explanation': explanation
            })
        
        return drivers
    
    def _format_driver_explanation(
        self,
        feature: str,
        value: any,
        direction: str,
        magnitude: float
    ) -> str:
        """Format driver into human-readable explanation"""
        
        # Mapping of features to explanations
        explanations = {
            'events_12m': f"{'Attended' if value > 0 else 'No'} {int(value)} events in last 12 months",
            'webinars_12m': f"{'Attended' if value > 0 else 'No'} {int(value)} webinars",
            'website_sessions_90d': f"{'Active' if value > 3 else 'Low'} website engagement ({int(value)} sessions)",
            'committee_member_flag': "Committee member" if value == 1 else "Not on any committee",
            'committee_leadership_flag': "Committee leadership role" if value == 1 else "No leadership role",
            'auto_renew_flag': "Auto-renew enabled" if value == 1 else "Auto-renew not enabled",
            'renewed_prev_year': "Renewed last year" if value == 1 else "Did not renew last year",
            'email_open_rate': f"Email engagement: {value*100:.0f}% open rate",
            'tenure_years': f"{int(value)} years of membership",
            'donation_amount_12m': f"{'Donated ${:.0f}'.format(value) if value > 0 else 'No donations'}",
            'zero_engagement': "No engagement activity" if value == 1 else "Some engagement",
            'total_engagement_score': f"Engagement score: {value:.1f}",
            'failed_payment_flag': "Payment failure" if value == 1 else "No payment issues"
        }
        
        base_explanation = explanations.get(feature, f"{feature}: {value}")
        impact_text = f"({'+' if direction == 'positive' else '-'}{magnitude:.1f}% impact)"
        
        return f"{base_explanation} {impact_text}"
    
    # ========== OUTPUT 4: Engagement Health Score ==========
    
    def calculate_engagement_score(
        self,
        member_data: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Output 4: Engagement Health Score (0-100)
        
        Composite score based on:
        - Events participation (25%)
        - Committee involvement (25%)
        - Website/portal activity (20%)
        - Email engagement (15%)
        - Leadership roles (15%)
        """
        scores = pd.DataFrame()
        scores['member_id'] = member_data['member_id']
        
        # Events participation score (0-25)
        max_events = member_data['events_12m'].quantile(0.95)
        scores['events_score'] = (member_data['events_12m'] / max_events * 25).clip(0, 25)
        
        # Committee involvement score (0-25)
        committee_score = (
            member_data['committee_member_flag'] * 15 +
            member_data['committee_leadership_flag'] * 10
        )
        scores['committee_score'] = committee_score.clip(0, 25)
        
        # Website/portal activity score (0-20)
        max_sessions = member_data['website_sessions_90d'].quantile(0.95)
        scores['portal_score'] = (member_data['website_sessions_90d'] / max_sessions * 20).clip(0, 20)
        
        # Email engagement score (0-15)
        email_score = (
            member_data['email_open_rate'] * 10 +
            member_data['email_click_rate'] * 5
        ) * 15
        scores['email_score'] = email_score.clip(0, 15)
        
        # Leadership roles score (0-15)
        leadership_score = member_data['committee_leadership_flag'] * 15
        scores['leadership_score'] = leadership_score.clip(0, 15)
        
        # Total engagement health score
        scores['engagement_health_score'] = (
            scores['events_score'] +
            scores['committee_score'] +
            scores['portal_score'] +
            scores['email_score'] +
            scores['leadership_score']
        ).round(1)
        
        # Calculate percentile rank
        scores['percentile_rank'] = scores['engagement_health_score'].rank(pct=True) * 100
        scores['percentile_rank'] = scores['percentile_rank'].round(0).astype(int)
        
        return scores[['member_id', 'engagement_health_score', 'percentile_rank']]
    
    # ========== OUTPUT 5: What-If Scenario Analysis ==========
    
    def simulate_intervention(
        self,
        member_data: pd.Series,
        current_probability: float,
        interventions: Dict[str, any],
        model,
        feature_names: List[str]
    ) -> Dict:
        """
        Output 5: What-If Scenario Analysis
        
        Simulate impact of interventions on renewal probability
        
        Interventions can include:
        - attend_event: Increase events_12m by N
        - join_committee: Set committee_member_flag = 1
        - portal_login: Increase website_sessions_90d by N
        - enable_auto_renew: Set auto_renew_flag = 1
        """
        # Create modified member data
        modified_data = member_data.copy()
        applied_interventions = []
        
        for intervention, value in interventions.items():
            if intervention == 'attend_event':
                modified_data['events_12m'] += value
                applied_interventions.append(f"Attend {value} event(s)")
                
            elif intervention == 'join_committee':
                modified_data['committee_member_flag'] = 1
                applied_interventions.append("Join a committee")
                
            elif intervention == 'portal_login':
                modified_data['website_sessions_90d'] += value
                applied_interventions.append(f"{value} additional portal logins")
                
            elif intervention == 'enable_auto_renew':
                modified_data['auto_renew_flag'] = 1
                applied_interventions.append("Enable auto-renew")
                
            elif intervention == 'attend_webinar':
                modified_data['webinars_12m'] += value
                applied_interventions.append(f"Attend {value} webinar(s)")
        
        # Recalculate derived features (simplified)
        modified_data['total_engagement_score'] = (
            modified_data['events_12m'] * 2.0 +
            modified_data['webinars_12m'] * 1.5 +
            modified_data['courses_12m'] * 2.5 +
            modified_data['website_sessions_90d'] * 0.5 +
            modified_data['community_logins_90d'] * 0.8
        )
        
        # Get new probability (would need actual model prediction)
        # For now, estimate based on engagement change
        engagement_delta = (
            modified_data['total_engagement_score'] - 
            member_data['total_engagement_score']
        )
        
        # Rough estimate: each 10 points of engagement = ~5% probability increase
        probability_delta = (engagement_delta / 10) * 0.05
        projected_probability = min(current_probability + probability_delta, 0.95)
        
        result = {
            'current_probability': round(current_probability * 100, 1),
            'projected_probability': round(projected_probability * 100, 1),
            'probability_change': round((projected_probability - current_probability) * 100, 1),
            'interventions_applied': applied_interventions,
            'recommendation': self._get_scenario_recommendation(
                current_probability, projected_probability
            )
        }
        
        return result
    
    def _get_scenario_recommendation(
        self,
        current_prob: float,
        projected_prob: float
    ) -> str:
        """Generate recommendation based on scenario results"""
        
        delta = projected_prob - current_prob
        
        if delta > 0.15:
            return "High impact intervention - Strongly recommended"
        elif delta > 0.08:
            return "Moderate impact - Recommended"
        elif delta > 0.03:
            return "Low impact - Consider other interventions"
        else:
            return "Minimal impact - Not recommended"
    
    # ========== OUTPUT 6: Executive Portfolio View ==========
    
    def generate_executive_summary(
        self,
        predictions_df: pd.DataFrame,
        member_data: pd.DataFrame
    ) -> Dict:
        """
        Output 6: Executive Portfolio View
        
        Aggregate statistics for leadership
        """
        probabilities = predictions_df['renewal_probability'].values / 100
        
        # Overall metrics
        total_members = len(predictions_df)
        high_risk_count = (predictions_df['risk_level'] == 'High Risk').sum()
        medium_risk_count = (predictions_df['risk_level'] == 'Medium Risk').sum()
        low_risk_count = (predictions_df['risk_level'] == 'Low Risk').sum()
        
        expected_renewals = int((probabilities >= 0.5).sum())
        expected_renewal_rate = (expected_renewals / total_members) * 100
        
        # Revenue at risk
        if 'dues_amount' in member_data.columns:
            high_risk_members = predictions_df[predictions_df['risk_level'] == 'High Risk']['member_id']
            revenue_at_risk = member_data[member_data['member_id'].isin(high_risk_members)]['dues_amount'].sum()
        else:
            revenue_at_risk = 0
        
        # Trends by segment
        trends = {}
        
        for segment_col in ['membership_type', 'chapter', 'tenure_years']:
            if segment_col in member_data.columns:
                merged = predictions_df.merge(
                    member_data[['member_id', segment_col]],
                    on='member_id',
                    how='left'
                )
                
                if segment_col == 'tenure_years':
                    # Group tenure into buckets
                    merged['segment'] = pd.cut(
                        merged[segment_col],
                        bins=[-1, 2, 5, 10, 100],
                        labels=['0-2 years', '3-5 years', '6-10 years', '10+ years']
                    )
                else:
                    merged['segment'] = merged[segment_col]
                
                segment_stats = merged.groupby('segment').agg({
                    'renewal_probability': ['mean', 'count']
                }).round(1)
                
                trends[segment_col] = segment_stats.to_dict()
        
        summary = {
            'overview': {
                'total_members': total_members,
                'expected_renewals': expected_renewals,
                'expected_renewal_rate': round(expected_renewal_rate, 1),
                'avg_renewal_probability': round(probabilities.mean() * 100, 1)
            },
            'risk_distribution': {
                'high_risk': {
                    'count': int(high_risk_count),
                    'percentage': round((high_risk_count / total_members) * 100, 1)
                },
                'medium_risk': {
                    'count': int(medium_risk_count),
                    'percentage': round((medium_risk_count / total_members) * 100, 1)
                },
                'low_risk': {
                    'count': int(low_risk_count),
                    'percentage': round((low_risk_count / total_members) * 100, 1)
                }
            },
            'financial_impact': {
                'revenue_at_risk': round(revenue_at_risk, 2),
                'high_risk_members': int(high_risk_count)
            }
        }
        
        return summary
    
    # ========== OUTPUT 7: CRM Action Recommendations ==========
    
    def generate_crm_actions(
        self,
        predictions_df: pd.DataFrame,
        member_data: pd.DataFrame = None
    ) -> pd.DataFrame:
        """
        Output 7: CRM-Ready Action Recommendations
        
        Generate personalized actions based on risk level
        """
        actions = predictions_df.copy()
        
        # Define action rules
        def get_action_details(row):
            risk = row['risk_level']
            prob = row['renewal_probability']
            
            if risk == 'High Risk':
                return {
                    'action_type': 'Personal Outreach',
                    'priority': 'High',
                    'channel': 'Phone Call + Email',
                    'message_template': 'personal_retention_call',
                    'recommended_offer': 'Discount or flexible payment',
                    'timeline': 'Immediate (within 7 days)'
                }
            elif risk == 'Medium Risk':
                return {
                    'action_type': 'Targeted Engagement',
                    'priority': 'Medium',
                    'channel': 'Email + Event Invitation',
                    'message_template': 'engagement_campaign',
                    'recommended_offer': 'Event invitation or webinar access',
                    'timeline': 'Within 14 days'
                }
            else:  # Low Risk
                return {
                    'action_type': 'Standard Renewal',
                    'priority': 'Low',
                    'channel': 'Email',
                    'message_template': 'standard_renewal_reminder',
                    'recommended_offer': 'None',
                    'timeline': 'Standard renewal cycle'
                }
        
        # Apply action rules
        action_details = actions.apply(get_action_details, axis=1, result_type='expand')
        actions = pd.concat([actions, action_details], axis=1)
        
        # Sort by priority
        priority_order = {'High': 1, 'Medium': 2, 'Low': 3}
        actions['priority_rank'] = actions['priority'].map(priority_order)
        actions = actions.sort_values(['priority_rank', 'renewal_probability'])
        actions = actions.drop('priority_rank', axis=1)
        
        return actions


if __name__ == "__main__":
    print("=" * 60)
    print("POC OUTPUT GENERATOR - TEST")
    print("=" * 60)
    
    # Create sample data
    np.random.seed(42)
    n_members = 100
    
    member_ids = [f"M{i:06d}" for i in range(1, n_members + 1)]
    probabilities = np.random.beta(2, 2, n_members)  # Sample probabilities
    
    # Initialize generator
    generator = POCOutputGenerator()
    
    # Test Output 1: Renewal Scores
    print("\n📊 Output 1: Renewal Probability Scores")
    scores = generator.generate_renewal_scores(pd.Series(member_ids), probabilities)
    print(scores.head(10))
    print(f"\nRisk Distribution:\n{scores['risk_level'].value_counts()}")
    
    # Test Output 2: Risk Segments
    print("\n📊 Output 2: Risk Segmentation")
    segments = generator.create_risk_segments(probabilities)
    for level, stats in segments.items():
        print(f"\n{level}: {stats}")
    
    print("\n✅ POC Output Generator tests complete!")
