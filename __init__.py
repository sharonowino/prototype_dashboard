"""
GTFS Disruption Detection Package
==================================

A modular pipeline for detecting and classifying transit disruptions
using GTFS (General Transit Feed Specification) data.

Modules
-------
- features: Feature engineering and disruption classification
- modeling: Temporal splitting, class balancing, and leakage detection
- evaluation: Metrics computation and visualization
- utils: Configuration and logging utilities
- pipeline: End-to-end orchestrator
- api: REST API for serving predictions

Usage
-----
>>> from gtfs_disruption import DisruptionPipeline
>>> pipeline = DisruptionPipeline("config.yaml")
>>> results = pipeline.run_full_pipeline(merged_df, gtfs_data)
"""

__version__ = "1.0.0"
__author__ = "GTFS Disruption Detection Team"

from .pipeline import DisruptionPipeline
from .features import DisruptionFeatureBuilder
from .features.classifier import DisruptionClassifier
from .features.analyzer import DisruptionAnalyzer
from .modeling import chronological_split, TemporalAwareBalancer, WalkForwardCV
from .modeling.leakage import detect_potential_leakage, verify_temporal_split
from .evaluation import compute_metrics, generate_classification_report
from .utils import load_config, setup_logging
from .api import DisruptionPredictor, PredictionAPI, create_api
from .ingestion import (
    ingest_local, ingest_live, ingest_combined,
    load_local_feeds, fetch_all_live_feeds, fetch_static_gtfs,
    load_static_gtfs_from_zip, merge_feed_data,
)

__all__ = [
    'DisruptionPipeline',
    'DisruptionFeatureBuilder',
    'DisruptionClassifier',
    'DisruptionAnalyzer',
    'chronological_split',
    'TemporalAwareBalancer',
    'WalkForwardCV',
    'detect_potential_leakage',
    'verify_temporal_split',
    'compute_metrics',
    'generate_classification_report',
    'load_config',
    'setup_logging',
    'DisruptionPredictor',
    'PredictionAPI',
    'create_api',
    'ingest_local',
    'ingest_live',
    'ingest_combined',
    'load_local_feeds',
    'fetch_all_live_feeds',
    'fetch_static_gtfs',
    'load_static_gtfs_from_zip',
    'merge_feed_data',
]
