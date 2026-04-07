"""
NLP Classification Module - L1 & L2 Intent Extraction
Uses Claude API for zero-shot classification of review intent and business impact
"""

import pandas as pd
import json
import logging
import time
from typing import List, Dict, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
import anthropic

from config import (
    NLP_CONFIG,
    L1_INTENT_CATEGORIES,
    SENTIMENT_LABELS,
    ANALYSIS_CONFIG,
    API_CONFIG,
)

logger = logging.getLogger(__name__)

# ============================================================================
# Intent Classification
# ============================================================================

class IntentClassifier:
    """
    Classify review intent using Claude API
    Provides both L1 (topic) and L2 (business impact) classifications
    """
    
    def __init__(self):
        """Initialize classifier with Anthropic client (falls back to keyword NLP if no API key)"""

        self.model = NLP_CONFIG["model"]
        self.temperature = NLP_CONFIG["temperature"]
        self.max_tokens = NLP_CONFIG["max_tokens"]
        self.timeout = NLP_CONFIG["timeout"]

        if API_CONFIG["anthropic_api_key"]:
            try:
                self.client = anthropic.Anthropic(api_key=API_CONFIG["anthropic_api_key"])
                self.api_available = True
                logger.info(f"Initialized classifier with model: {self.model}")
            except Exception as e:
                logger.warning(f"Failed to initialise Anthropic client: {e}. Using keyword classifier.")
                self.client = None
                self.api_available = False
        else:
            logger.info("No ANTHROPIC_API_KEY found — using enhanced keyword-based classifier.")
            self.client = None
            self.api_available = False
    
    def classify_review(self, text: str, rating: int) -> Dict:
        """
        Classify a single review for L1 and L2 intent
        
        Args:
            text: Review text
            rating: Star rating (1-5)
            
        Returns:
            Dict with classification results
        """
        
        # Skip API entirely if not available
        if not self.api_available:
            return self._get_fallback_classification(text, rating)

        try:
            # System prompt for classification
            system_prompt = """You are an expert at understanding customer intent from app reviews.

Your task: Classify this Royal Enfield app review into:
1. L1 Intent: Primary topic category
2. L2 Intent: Business impact area
3. Sentiment: Overall tone

Available L1 categories: Performance, Features, UI/UX, Customer_Support, Bugs, Billing, Security, Other

Available L2 Impact areas:
- User Acquisition: affects new user conversion
- User Retention: affects churn rate
- Engagement: affects time in app, feature usage
- Brand Trust: affects reputation, NPS
- Revenue: affects direct revenue/refunds
- Product Roadmap: guides feature development

Return ONLY valid JSON with this exact structure:
{
  "l1_intent": "category_name",
  "l1_confidence": 0.95,
  "l2_impact": "impact_area_name",
  "l2_confidence": 0.88,
  "sentiment": "positive|negative|neutral",
  "summary": "brief 1-line summary"
}"""

            user_message = f"""Rating: {rating}/5
Review: "{text}"

Classify this review."""

            # Call Claude API
            response = self.client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                system=system_prompt,
                messages=[
                    {"role": "user", "content": user_message}
                ],
                timeout=self.timeout,
            )
            
            # Parse response
            response_text = response.content[0].text
            classification = json.loads(response_text)
            
            # Validate response
            classification = self._validate_classification(classification)
            
            return {
                "text": text[:100],  # First 100 chars for reference
                "rating": rating,
                **classification,
                "raw_response": response_text,
            }
        
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON response: {e}")
            return self._get_fallback_classification(text, rating)
        
        except anthropic.APIError as e:
            logger.error(f"API error: {e}")
            return self._get_fallback_classification(text, rating)
    
    def classify_batch(self, df: pd.DataFrame, batch_size: int = 10, 
                      max_workers: int = 3) -> pd.DataFrame:
        """
        Classify multiple reviews with parallel processing
        
        Args:
            df: DataFrame with 'text' and 'rating' columns
            batch_size: Reviews per API batch
            max_workers: Number of parallel workers
            
        Returns:
            DataFrame with classification results
        """
        
        logger.info(f"Starting batch classification of {len(df)} reviews...")
        
        results = []
        total = len(df)
        
        # Process with rate limiting
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = []
            
            for idx, row in df.iterrows():
                future = executor.submit(
                    self.classify_review,
                    text=row['text'],
                    rating=int(row['rating'])
                )
                futures.append((idx, future))
                
                # Print progress every 10 reviews
                if (idx + 1) % 10 == 0:
                    logger.info(f"Processing... {idx + 1}/{total}")
            
            # Collect results as they complete
            for idx, future in futures:
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    logger.error(f"Error processing review {idx}: {e}")
                    results.append(self._get_fallback_classification(
                        df.iloc[idx]['text'],
                        int(df.iloc[idx]['rating'])
                    ))
        
        # Convert to DataFrame
        df_classified = pd.DataFrame(results)
        
        logger.info(f"✅ Batch classification complete. {len(df_classified)} reviews classified")
        
        return df_classified
    
    @staticmethod
    def _validate_classification(classification: Dict) -> Dict:
        """Validate and clean classification response"""
        
        # Set defaults if missing
        classification.setdefault('l1_intent', 'Other')
        classification.setdefault('l1_confidence', 0.5)
        classification.setdefault('l2_impact', 'Engagement')
        classification.setdefault('l2_confidence', 0.5)
        classification.setdefault('sentiment', 'neutral')
        classification.setdefault('summary', '')
        
        # Ensure confidences are floats between 0-1
        classification['l1_confidence'] = float(classification.get('l1_confidence', 0.5))
        classification['l2_confidence'] = float(classification.get('l2_confidence', 0.5))
        classification['l1_confidence'] = min(1.0, max(0.0, classification['l1_confidence']))
        classification['l2_confidence'] = min(1.0, max(0.0, classification['l2_confidence']))
        
        # Ensure sentiment is valid
        valid_sentiments = ['positive', 'negative', 'neutral']
        if classification['sentiment'] not in valid_sentiments:
            classification['sentiment'] = 'neutral'
        
        return classification
    
    @staticmethod
    def _get_fallback_classification(text: str, rating: int) -> Dict:
        """Return fallback classification if API fails"""
        
        # Determine sentiment from rating
        if rating >= 4:
            sentiment = 'positive'
        elif rating <= 2:
            sentiment = 'negative'
        else:
            sentiment = 'neutral'
        
        # Enhanced keyword-based L1 classification
        # Keywords are ordered by specificity — first match wins
        text_lower = text.lower()

        KEYWORD_MAP = [
            ('Security',          ['privacy', 'data', 'hack', 'secure', 'otp', 'password', 'access', 'leak', 'stolen', 'unauthorised', 'unauthorized']),
            ('Billing',           ['price', 'payment', 'billing', 'cost', 'refund', 'charge', 'subscription', 'purchase', 'money', 'paid', 'fee', 'transaction']),
            ('Performance',       ['crash', 'crashes', 'crashing', 'slow', 'lag', 'lagging', 'hang', 'hangs', 'hanging', 'battery', 'freeze', 'freezing',
                                   'loading', 'speed', 'unresponsive', 'stuttering', 'stuck', 'not opening', 'won\'t open', 'won\'t load',
                                   'keeps stopping', 'force close', 'force stop', 'black screen', 'white screen']),
            ('Bugs',              ['bug', 'bugs', 'error', 'errors', 'problem', 'problems', 'issue', 'issues', 'fix', 'broken', 'glitch',
                                   'not working', 'doesn\'t work', 'does not work', 'fail', 'fails', 'failed', 'login', 'otp not',
                                   'cannot login', 'can\'t login', 'sign in', 'blank', 'malfunction', 'incorrect']),
            ('Customer_Support',  ['support', 'customer care', 'service', 'delivery', 'dealer', 'helpline', 'response', 'waiting', 'wait',
                                   'no reply', 'no response', 'staff', 'agent', 'complaint', 'escalate', 'after sales', 'warranty',
                                   'service center', 'rto', 'showroom']),
            ('UI/UX',             ['ui', 'design', 'interface', 'confusing', 'layout', 'button', 'navigation', 'screen', 'dark mode',
                                   'theme', 'look', 'ugly', 'clunky', 'cluttered', 'complicated', 'hard to use', 'easy to use',
                                   'user friendly', 'not intuitive', 'font', 'colour', 'color', 'icon']),
            ('Features',          ['feature', 'features', 'missing', 'add', 'configurator', 'wishlist', 'request', 'booking',
                                   'trip', 'route', 'map', 'tracker', 'gps', 'service history', 'insurance', 'emi', 'calculator',
                                   'garage', 'community', 'forum', 'notification', 'update', 'option', 'need', 'want', 'should have']),
        ]

        l1_intent = 'Other'
        for category, keywords in KEYWORD_MAP:
            if any(kw in text_lower for kw in keywords):
                l1_intent = category
                break
        
        # Map to L2 impact
        l2_map = {
            'Performance': 'User Retention',
            'Features': 'Engagement',
            'UI/UX': 'Engagement',
            'Customer_Support': 'Brand Trust',
            'Bugs': 'User Retention',
            'Billing': 'Revenue',
            'Security': 'Brand Trust',
            'Other': 'Engagement',
        }
        
        return {
            "text": text[:100],
            "rating": rating,
            "l1_intent": l1_intent,
            "l1_confidence": 0.6,
            "l2_impact": l2_map[l1_intent],
            "l2_confidence": 0.5,
            "sentiment": sentiment,
            "summary": f"{sentiment} {l1_intent} review",
            "raw_response": "fallback_classification",
        }


# ============================================================================
# Sentiment Analysis Utility
# ============================================================================

class SentimentAnalyzer:
    """Analyze sentiment from rating and text"""
    
    @staticmethod
    def get_sentiment_label(rating: int) -> str:
        """Get sentiment label from rating"""
        return SENTIMENT_LABELS.get(rating, "neutral")
    
    @staticmethod
    def analyze_batch(df: pd.DataFrame) -> pd.DataFrame:
        """Analyze sentiment for batch of reviews"""
        
        df['sentiment_label'] = df['rating'].apply(SentimentAnalyzer.get_sentiment_label)
        
        # Map to simple positive/negative/neutral
        def map_sentiment(label):
            if label in ['very_positive', 'positive']:
                return 'positive'
            elif label in ['very_negative', 'negative']:
                return 'negative'
            else:
                return 'neutral'
        
        df['sentiment_simple'] = df['sentiment_label'].apply(map_sentiment)
        
        return df


# ============================================================================
# Main Execution
# ============================================================================

def classify_reviews(df: pd.DataFrame) -> pd.DataFrame:
    """
    Classify all reviews
    
    Args:
        df: DataFrame with reviews
        
    Returns:
        DataFrame with classification results
    """
    
    logger.info(f"Starting classification of {len(df)} reviews...")
    
    classifier = IntentClassifier()
    df_classified = classifier.classify_batch(df)
    
    # Add sentiment analysis
    df_classified = SentimentAnalyzer.analyze_batch(df_classified)
    
    logger.info("✅ Classification complete")
    
    return df_classified


if __name__ == "__main__":
    # Test classification
    logging.basicConfig(level=logging.INFO)
    
    # Test review
    test_df = pd.DataFrame({
        'text': [
            "App keeps crashing on login. Very frustrating!",
            "Love the bike configurator feature. Amazing!",
            "Decent app but support response is slow.",
        ],
        'rating': [1, 5, 3]
    })
    
    result = classify_reviews(test_df)
    print(result[['text', 'rating', 'l1_intent', 'l2_impact', 'sentiment']])
