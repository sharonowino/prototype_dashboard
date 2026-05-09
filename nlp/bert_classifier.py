"""
Enhanced NLP Module with BERT-based Alert Classification
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
import streamlit as st

# Global BERT classifier instance
_bert_classifier = None
_classification_cache = {}

def load_bert_classifier():
    """Load or initialize BERT classifier for alert categorization."""
    global _bert_classifier
    
    if not BERT_AVAILABLE:
        return None
    
    try:
        # Try loading fine-tuned model if exists
        model_path = Path(__file__).parent.parent / 'models' / 'bert_alert_classifier'
        if model_path.exists():
            classifier = pipeline(
                "text-classification",
                model=str(model_path),
                tokenizer=str(model_path),
                device=0 if torch.cuda.is_available() else -1
            )
        else:
            # Fallback: keyword-based classifier (rule-based)
            classifier = None
        
        _bert_classifier = classifier
        return classifier
    except Exception as e:
        st.warning(f"BERT load failed: {e}, using keyword fallback")
        return None

def classify_alert_bert_batch(texts: List[str], batch_size: int = 32) -> List[Dict[str, float]]:
    """Batch classify multiple alert texts."""
    if not _bert_classifier:
        return [keyword_classify(txt) for txt in texts]
    
    results = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        try:
            predictions = _bert_classifier(batch, truncation=True, max_length=128)
            for pred in predictions:
                # Convert to taxonomy format
                label = pred['label']
                score = pred['score']
                results.append({label: score})
        except Exception as e:
            results.extend([keyword_classify(txt) for txt in batch])
    
    return results

def keyword_classify(text: str) -> Dict[str, float]:
    """Keyword-based fallback classification."""
    from collections import defaultdict
    
    if not text or pd.isna(text):
        return {"UNKNOWN": 1.0}
    
    text_lower = str(text).lower()
    scores = defaultdict(float)
    
    for category, keywords in ALERT_TAXONOMY.items():
        matched = sum(1 for kw in keywords if kw in text_lower)
        if matched > 0:
            scores[category] = matched / len(keywords)
    
    if not scores:
        scores["UNKNOWN"] = 1.0
    
    # Normalize
    total = sum(scores.values())
    return {k: v/total for k, v in scores.items()}

def get_primary_alert_category(text: str) -> tuple:
    """Get primary category and confidence for alert text."""
    # Check cache first
    cache_key = str(text)[:200]
    if cache_key in _classification_cache:
        result = _classification_cache[cache_key]
    else:
        classifier = load_bert_classifier()
        if classifier:
            try:
                preds = classifier(str(text)[:512], truncation=True)
                result = {preds[0]['label']: preds[0]['score']}
            except:
                result = keyword_classify(text)
        else:
            result = keyword_classify(text)
        _classification_cache[cache_key] = result
    
    primary = max(result, key=result.get)
    confidence = result[primary]
    return primary, confidence

def add_bert_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add BERT classification features to DataFrame."""
    df = df.copy()
    
    if 'description_text' not in df.columns:
        # No alert text available
        df['alert_primary_category'] = 'UNKNOWN'
        df['alert_category_confidence'] = 0.0
        for cat in ALERT_TAXONOMY.keys():
            df[f'alert_category_{cat}'] = 0
        return df
    
    # Initialize classifier on first use
    if 'bert_classifier_loaded' not in st.session_state:
        load_bert_classifier()
        st.session_state.bert_classifier_loaded = True
    
    # Apply classification
    categories = []
    confidences = []
    category_scores = {cat: [] for cat in ALERT_TAXONOMY.keys()}
    
    for text in df['description_text'].fillna(''):
        primary, conf = get_primary_alert_category(text)
        categories.append(primary)
        confidences.append(conf)
        
        # Get full result for one-hot encoding
        full_result = keyword_classify(text)  # Simplified; BERT would give all scores
        for cat in category_scores:
            category_scores[cat].append(full_result.get(cat, 0.0))
    
    df['alert_primary_category'] = categories
    df['alert_category_confidence'] = confidences
    
    # One-hot encode top categories
    for cat in ALERT_TAXONOMY.keys():
        df[f'alert_category_{cat}'] = category_scores[cat]
    
    return df

def clear_classification_cache():
    """Clear BERT classification cache."""
    global _classification_cache
    _classification_cache = {}
