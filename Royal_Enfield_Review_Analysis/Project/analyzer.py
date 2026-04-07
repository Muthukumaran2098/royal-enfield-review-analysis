"""
Analysis Module - Exploratory Data Analysis & Insights Generation
Produces key metrics, visualizations, and actionable recommendations
"""

import pandas as pd
import numpy as np
import logging
from datetime import datetime
from typing import Dict, List, Tuple, Optional

from config import (
    ANALYSIS_CONFIG,
    PRIORITIZATION_CONFIG,
    OUTPUT_DIR,
)

logger = logging.getLogger(__name__)

# ============================================================================
# Exploratory Data Analysis (EDA)
# ============================================================================

class ExploratoryAnalysis:
    """Perform EDA on review data"""
    
    @staticmethod
    def analyze(df: pd.DataFrame) -> Dict:
        """
        Perform comprehensive EDA
        
        Args:
            df: DataFrame with reviews and classifications
            
        Returns:
            Dictionary with EDA results
        """
        
        logger.info("Starting exploratory data analysis...")
        
        results = {
            "dataset_overview": ExploratoryAnalysis._dataset_overview(df),
            "rating_analysis": ExploratoryAnalysis._rating_analysis(df),
            "temporal_analysis": ExploratoryAnalysis._temporal_analysis(df),
            "text_analysis": ExploratoryAnalysis._text_analysis(df),
        }
        
        logger.info("✅ EDA complete")
        
        return results
    
    @staticmethod
    def _dataset_overview(df: pd.DataFrame) -> Dict:
        """Dataset overview statistics"""
        
        return {
            "total_reviews": len(df),
            "date_range": {
                "start": str(df['date'].min()) if 'date' in df.columns else None,
                "end": str(df['date'].max()) if 'date' in df.columns else None,
                "days": (df['date'].max() - df['date'].min()).days if 'date' in df.columns else None,
            },
            "data_quality": {
                "missing_values": df.isnull().sum().to_dict(),
                "percent_complete": ((1 - df.isnull().sum() / len(df)) * 100).round(1).to_dict(),
            },
        }
    
    @staticmethod
    def _rating_analysis(df: pd.DataFrame) -> Dict:
        """Rating distribution and statistics"""
        
        if 'rating' not in df.columns:
            return {}
        
        rating_counts = df['rating'].value_counts().sort_index(ascending=False)
        rating_pct = (rating_counts / len(df) * 100).round(1)
        
        return {
            "mean_rating": float(df['rating'].mean()),
            "median_rating": float(df['rating'].median()),
            "std_dev": float(df['rating'].std()),
            "distribution": {
                f"{int(stars)}_star": {
                    "count": int(rating_counts.get(stars, 0)),
                    "percent": float(rating_pct.get(stars, 0)),
                }
                for stars in range(5, 0, -1)
            },
            "segments": {
                "positive_4_5": {
                    "count": len(df[df['rating'] >= 4]),
                    "percent": round(len(df[df['rating'] >= 4]) / len(df) * 100, 1),
                },
                "neutral_3": {
                    "count": len(df[df['rating'] == 3]),
                    "percent": round(len(df[df['rating'] == 3]) / len(df) * 100, 1),
                },
                "negative_1_2": {
                    "count": len(df[df['rating'] <= 2]),
                    "percent": round(len(df[df['rating'] <= 2]) / len(df) * 100, 1),
                },
            },
        }
    
    @staticmethod
    def _temporal_analysis(df: pd.DataFrame) -> Dict:
        """Temporal trends in reviews"""
        
        if 'date' not in df.columns:
            return {}
        
        # Monthly trends
        df_copy = df.copy()
        df_copy['month'] = pd.to_datetime(df_copy['date']).dt.to_period('M')
        
        monthly_stats = []
        for month in sorted(df_copy['month'].unique(), reverse=True)[:6]:  # Last 6 months
            month_data = df_copy[df_copy['month'] == month]
            monthly_stats.append({
                "month": str(month),
                "review_count": len(month_data),
                "avg_rating": round(month_data['rating'].mean(), 2) if 'rating' in month_data.columns else None,
            })
        
        return {
            "monthly_trends": monthly_stats,
            "trend": "improving" if monthly_stats[0]['avg_rating'] > monthly_stats[-1]['avg_rating'] 
                     else "declining",
        }
    
    @staticmethod
    def _text_analysis(df: pd.DataFrame) -> Dict:
        """Text statistics"""
        
        if 'text' not in df.columns:
            return {}
        
        df['word_count'] = df['text'].str.split().str.len()
        df['char_count'] = df['text'].str.len()
        
        return {
            "avg_review_length_chars": int(df['char_count'].mean()),
            "avg_review_length_words": int(df['word_count'].mean()),
            "min_length_chars": int(df['char_count'].min()),
            "max_length_chars": int(df['char_count'].max()),
            "correlation_length_vs_rating": round(df['word_count'].corr(df['rating']), 3),
        }


# ============================================================================
# Intent Analysis
# ============================================================================

class IntentAnalysis:
    """Analyze L1 and L2 intent distributions"""
    
    @staticmethod
    def analyze_l1(df: pd.DataFrame) -> Dict:
        """Analyze L1 intent (topics)"""
        
        if 'l1_intent' not in df.columns:
            return {}
        
        logger.info("Analyzing L1 intent distribution...")
        
        l1_counts = df['l1_intent'].value_counts()
        l1_pct = (l1_counts / len(df) * 100).round(1)
        
        # Sentiment by topic
        topic_sentiment = {}
        for topic in df['l1_intent'].unique():
            topic_df = df[df['l1_intent'] == topic]
            topic_sentiment[topic] = {
                "count": len(topic_df),
                "avg_rating": round(topic_df['rating'].mean(), 2),
                "sentiment_dist": topic_df['sentiment_simple'].value_counts().to_dict() if 'sentiment_simple' in topic_df.columns else {},
            }
        
        return {
            "topic_distribution": {
                topic: {
                    "count": int(count),
                    "percent": float(pct),
                }
                for topic, (count, pct) in zip(l1_counts.index, zip(l1_counts.values, l1_pct.values))
            },
            "topic_sentiment": topic_sentiment,
        }
    
    @staticmethod
    def analyze_l2(df: pd.DataFrame) -> Dict:
        """Analyze L2 intent (business impact)"""
        
        if 'l2_impact' not in df.columns:
            return {}
        
        logger.info("Analyzing L2 impact distribution...")
        
        l2_counts = df['l2_impact'].value_counts()
        l2_pct = (l2_counts / len(df) * 100).round(1)
        
        # Impact severity
        impact_severity = {}
        for impact in df['l2_impact'].unique():
            impact_df = df[df['l2_impact'] == impact]
            impact_severity[impact] = {
                "count": len(impact_df),
                "avg_rating": round(impact_df['rating'].mean(), 2),
                "negative_percent": round(len(impact_df[impact_df['rating'] <= 2]) / len(impact_df) * 100, 1),
            }
        
        return {
            "impact_distribution": {
                impact: {
                    "count": int(count),
                    "percent": float(pct),
                }
                for impact, (count, pct) in zip(l2_counts.index, zip(l2_counts.values, l2_pct.values))
            },
            "impact_severity": impact_severity,
        }


# ============================================================================
# Prioritization Matrix
# ============================================================================

class PrioritizationMatrix:
    """Calculate priority scores for recommendations"""
    
    @staticmethod
    def calculate_priorities(df: pd.DataFrame) -> List[Dict]:
        """
        Calculate priority scores for each topic using impact matrix
        
        Args:
            df: DataFrame with classified reviews
            
        Returns:
            List of prioritized recommendations
        """
        
        logger.info("Calculating prioritization matrix...")
        
        if 'l1_intent' not in df.columns:
            return []
        
        priorities = []
        
        for topic in df['l1_intent'].unique():
            topic_df = df[df['l1_intent'] == topic]
            
            # Frequency score (0-100)
            frequency = (len(topic_df) / len(df)) * 100
            
            # Impact score based on negative sentiment (0-100)
            negative_count = len(topic_df[topic_df['rating'] <= 2])
            avg_rating = topic_df['rating'].mean()
            impact_score = (5 - avg_rating) * 20  # Scale to 0-100
            
            # Combine scores using weights
            priority_score = (
                frequency * PRIORITIZATION_CONFIG["frequency_weight"] +
                impact_score * PRIORITIZATION_CONFIG["impact_weight"]
            )
            
            # Determine priority level
            if priority_score >= PRIORITIZATION_CONFIG["p0_threshold"]:
                priority = "P0"
                priority_label = "Critical"
            elif priority_score >= PRIORITIZATION_CONFIG["p1_threshold"]:
                priority = "P1"
                priority_label = "High"
            elif priority_score >= PRIORITIZATION_CONFIG["p2_threshold"]:
                priority = "P2"
                priority_label = "Medium"
            else:
                priority = "P3"
                priority_label = "Low"
            
            priorities.append({
                "topic": topic,
                "frequency": round(frequency, 1),
                "negative_count": int(negative_count),
                "avg_rating": round(avg_rating, 2),
                "impact_score": round(impact_score, 1),
                "priority_score": round(priority_score, 1),
                "priority": priority,
                "priority_label": priority_label,
                "sample_reviews": topic_df.nsmallest(2, 'rating')['text'].tolist(),
            })
        
        # Sort by priority score
        priorities.sort(key=lambda x: x['priority_score'], reverse=True)
        
        logger.info(f"✅ Generated {len(priorities)} priority items")
        
        return priorities


# ============================================================================
# Insights Generation
# ============================================================================

class InsightsGenerator:
    """Generate actionable insights from analysis"""
    
    @staticmethod
    def generate(df: pd.DataFrame, eda_results: Dict, priorities: List[Dict]) -> Dict:
        """
        Generate comprehensive insights
        
        Args:
            df: DataFrame with reviews
            eda_results: EDA results
            priorities: Priority matrix results
            
        Returns:
            Dictionary with insights
        """
        
        logger.info("Generating insights...")
        
        insights = {
            "key_metrics": {
                "total_reviews": len(df),
                "avg_rating": round(df['rating'].mean(), 2),
                "rating_trend": eda_results.get("temporal_analysis", {}).get("trend", "stable"),
                "negative_reviews": len(df[df['rating'] <= 2]),
            },
            "top_issues": InsightsGenerator._get_top_issues(priorities),
            "strengths": InsightsGenerator._get_strengths(df),
            "recommendations": InsightsGenerator._get_recommendations(priorities),
        }
        
        logger.info("✅ Insights generated")
        
        return insights
    
    @staticmethod
    def _get_top_issues(priorities: List[Dict]) -> List[Dict]:
        """Get top negative topics"""
        
        top_issues = []
        for item in priorities[:3]:
            if item['avg_rating'] < 3:
                top_issues.append({
                    "topic": item['topic'],
                    "impact": item['priority_label'],
                    "negative_percent": round(item['negative_count'] / item['frequency'], 1),
                    "sample": item['sample_reviews'][0] if item['sample_reviews'] else "No samples",
                })
        
        return top_issues
    
    @staticmethod
    def _get_strengths(df: pd.DataFrame) -> List[Dict]:
        """Get positive topics from reviews"""
        
        if 'l1_intent' not in df.columns:
            return []
        
        strengths = []
        for topic in df['l1_intent'].unique():
            topic_df = df[df['l1_intent'] == topic]
            avg_rating = topic_df['rating'].mean()
            
            if avg_rating >= 4:
                positive_reviews = topic_df[topic_df['rating'] >= 4]
                strengths.append({
                    "topic": topic,
                    "avg_rating": round(avg_rating, 2),
                    "positive_count": len(positive_reviews),
                    "sample": positive_reviews['text'].iloc[0] if len(positive_reviews) > 0 else "",
                })
        
        return sorted(strengths, key=lambda x: x['avg_rating'], reverse=True)[:3]
    
    @staticmethod
    def _get_recommendations(priorities: List[Dict]) -> List[Dict]:
        """Generate actionable recommendations"""
        
        recommendations = []
        
        for item in priorities:
            if item['priority'] in ['P0', 'P1']:
                recommendation = {
                    "priority": item['priority'],
                    "action": f"Address {item['topic']} issues",
                    "rationale": f"{item['frequency']}% of reviews mention {item['topic'].lower()} with avg rating {item['avg_rating']}/5",
                    "timeline": "2 weeks" if item['priority'] == 'P0' else "4-6 weeks",
                    "impact": f"Expected to improve {item['negative_count']} negative reviews",
                }
                recommendations.append(recommendation)
        
        return recommendations


# ============================================================================
# Main Analysis Pipeline
# ============================================================================

def run_analysis(df: pd.DataFrame) -> Dict:
    """
    Run complete analysis pipeline
    
    Args:
        df: DataFrame with reviews and classifications
        
    Returns:
        Dictionary with all analysis results
    """
    
    logger.info("="*70)
    logger.info("STARTING COMPREHENSIVE ANALYSIS")
    logger.info("="*70)
    
    # EDA
    eda_results = ExploratoryAnalysis.analyze(df)
    
    # Intent Analysis
    l1_analysis = IntentAnalysis.analyze_l1(df)
    l2_analysis = IntentAnalysis.analyze_l2(df)
    
    # Prioritization
    priorities = PrioritizationMatrix.calculate_priorities(df)
    
    # Insights
    insights = InsightsGenerator.generate(df, eda_results, priorities)
    
    # Compile all results
    analysis_results = {
        "timestamp": datetime.now().isoformat(),
        "eda": eda_results,
        "l1_analysis": l1_analysis,
        "l2_analysis": l2_analysis,
        "priorities": priorities,
        "insights": insights,
    }
    
    logger.info("="*70)
    logger.info("✅ ANALYSIS COMPLETE")
    logger.info("="*70)
    
    return analysis_results


if __name__ == "__main__":
    # Test
    logging.basicConfig(level=logging.INFO)
    
    import pandas as pd
    
    # Create test data
    test_df = pd.DataFrame({
        'text': ['Great app!', 'Crashes constantly', 'Love the features'],
        'rating': [5, 1, 4],
        'l1_intent': ['Features', 'Performance', 'Features'],
        'l2_impact': ['Engagement', 'User Retention', 'Engagement'],
        'sentiment_simple': ['positive', 'negative', 'positive'],
        'date': pd.date_range('2024-01-01', periods=3),
    })
    
    results = run_analysis(test_df)
    print("\nAnalysis Results:")
    print(f"Average Rating: {results['insights']['key_metrics']['avg_rating']}")
    print(f"Top Issues: {len(results['insights']['top_issues'])}")
