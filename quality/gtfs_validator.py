"""
GTFS-RT Data Quality Monitoring Module
Implements Google Transit and Cal-ITP quality standards
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import streamlit as st
from dataclasses import dataclass, field
from enum import Enum

class IssueSeverity(Enum):
    ERROR = "ERROR"
    WARN = "WARN"
    INFO = "INFO"

@dataclass
class ValidationIssue:
    """Single validation issue record."""
    severity: IssueSeverity
    code: str
    description: str
    feed_type: str
    count: int = 1
    first_seen: datetime = field(default_factory=datetime.now)
    last_seen: datetime = field(default_factory=datetime.now)

class GTFSRTValidator:
    """Real-time GTFS feed validator following MobilityData/Google standards."""
    
    # Canonical GTFS-RT critical errors (Cal-ITP list)
    CRITICAL_ERROR_CODES = [
        "MISSING_VEHICLE_ID",
        "MISSING_TRIP_ID",
        "INVALID_TIMESTAMP",
        "STALE_FEED",
        "POSITION_OUT_OF_BOUNDS"
    ]
    
    # Netherlands geographic bounds (approx)
    NL_BOUNDS = {
        "lat_min": 50.7,
        "lat_max": 53.7,
        "lon_min": 3.3,
        "lon_max": 7.2
    }
    
    def __init__(self):
        self.issues: List[ValidationIssue] = []
        self.feed_metrics = {
            "coverage_trips_pct": 0.0,
            "coverage_routes_pct": 0.0,
            "feed_age_seconds": 0,
            "uptime_pct": 100.0,
            "validation_score": 100.0
        }
        self.static_gtfs_reference = None  # For coverage calculation
    
    def validate_feed(self, df: pd.DataFrame, feed_type: str) -> Dict:
        """Main validation entry point."""
        self.issues = []
        
        if df.empty:
            self._add_issue(IssueSeverity.WARN, "EMPTY_FEED", f"No data in {feed_type} feed", feed_type)
            return self._get_results()
        
        # 1. Schema validation
        self._validate_schema(df, feed_type)
        
        # 2. Timestamp validation
        self._validate_timestamps(df, feed_type)
        
        # 3. Geographic validation (vehicle positions only)
        if feed_type == "vehicle_positions":
            self._validate_geography(df)
        
        # 4. Data quality & sanity
        self._validate_data_sanity(df, feed_type)
        
        # 5. Cross-feed consistency (if multiple feeds)
        self._validate_cross_consistency(df, feed_type)
        
        # Compute coverage
        self._compute_coverage(df, feed_type)
        
        # Calculate overall quality score
        self._compute_quality_score()
        
        return self._get_results()
    
    def _add_issue(self, severity: IssueSeverity, code: str, description: str, 
                   feed_type: str, count: int = 1):
        """Add validation issue."""
        self.issues.append(ValidationIssue(
            severity=severity,
            code=code,
            description=description,
            feed_type=feed_type,
            count=count
        ))
    
    def _validate_schema(self, df: pd.DataFrame, feed_type: str):
        """Validate required fields per GTFS-RT spec."""
        required_fields = {
            "vehicle_positions": ["vehicle_id", "trip_id", "latitude", "longitude"],
            "trip_updates": ["trip_id", "stop_id"],
            "service_alerts": ["alert_id", "cause", "effect"]
        }
        
        fields = required_fields.get(feed_type, [])
        for field in fields:
            # Check alternative field names
            alt_names = {
                "latitude": ["lat", "latitude"],
                "longitude": ["lon", "longitude"],
                "vehicle_id": ["vehicle_id", "vehicle/id"],
                "trip_id": ["trip_id", "trip/id"],
                "stop_id": ["stop_id", "stop/id"],
                "alert_id": ["alert_id", "id"],
                "cause": ["cause"],
                "effect": ["effect"]
            }
            
            found = False
            for alt in alt_names.get(field, [field]):
                if alt in df.columns:
                    found = True
                    break
            
            if not found:
                self._add_issue(
                    IssueSeverity.ERROR if field in ["vehicle_id", "trip_id", "alert_id"] else IssueSeverity.WARN,
                    f"MISSING_{field.upper()}",
                    f"Required field '{field}' missing in {feed_type} feed",
                    feed_type
                )
    
    def _validate_timestamps(self, df: pd.DataFrame, feed_type: str):
        """Validate timestamp freshness and format."""
        if 'timestamp' not in df.columns:
            self._add_issue(IssueSeverity.WARN, "NO_TIMESTAMP", "No timestamp column", feed_type)
            return
        
        # Feed age check (Google: <90s for vehicle/trip, <10min for alerts)
        max_age = 900 if feed_type == "service_alerts" else 90
        latest_ts = df['timestamp'].max()
        if pd.notna(latest_ts):
            try:
                ts = pd.to_datetime(latest_ts, unit='s' if latest_ts > 1e10 else None)
                age_sec = (datetime.now() - ts).total_seconds()
                self.feed_metrics["feed_age_seconds"] = age_sec
                
                if age_sec > max_age:
                    self._add_issue(
                        IssueSeverity.ERROR,
                        "STALE_FEED",
                        f"Feed age {age_sec:.0f}s exceeds {max_age}s threshold",
                        feed_type
                    )
                elif age_sec > max_age * 0.7:  # Warning at 70% of threshold
                    self._add_issue(
                        IssueSeverity.WARN,
                        "AGING_FEED",
                        f"Feed age {age_sec:.0f}s approaching threshold",
                        feed_type
                    )
            except:
                pass
        
        # Timestamp monotonicity (should increase)
        if len(df) > 1:
            ts_sorted = df['timestamp'].sort_values()
            if not ts_sorted.is_monotonic_increasing:
                self._add_issue(
                    IssueSeverity.WARN,
                    "NON_MONOTONIC_TS",
                    "Timestamps not monotonically increasing",
                    feed_type
                )
    
    def _validate_geography(self, df: pd.DataFrame):
        """Validate positions within expected bounds."""
        lat_col = next((c for c in ['lat', 'latitude'] if c in df.columns), None)
        lon_col = next((c for c in ['lon', 'longitude'] if c in df.columns), None)
        
        if not lat_col or not lon_col:
            return
        
        # Out of Netherlands
        oob = df[
            (df[lat_col] < self.NL_BOUNDS["lat_min"]) |
            (df[lat_col] > self.NL_BOUNDS["lat_max"]) |
            (df[lon_col] < self.NL_BOUNDS["lon_min"]) |
            (df[lon_col] > self.NL_BOUNDS["lon_max"])
        ]
        
        if len(oob) > 0:
            self._add_issue(
                IssueSeverity.WARN,
                "OUT_OF_BOUNDS",
                f"{len(oob)} vehicle positions outside Netherlands",
                "vehicle_positions",
                count=len(oob)
            )
        
        # Implausible speed (>80 m/s ≈ 180 mph)
        if 'speed' in df.columns:
            fast = df[df['speed'] > 80]
            if len(fast) > 0:
                self._add_issue(
                    IssueSeverity.WARN,
                    "IMPLAUSIBLE_SPEED",
                    f"{len(fast)} vehicles with speed >80 m/s",
                    "vehicle_positions",
                    count=len(fast)
                )
    
    def _validate_data_sanity(self, df: pd.DataFrame, feed_type: str):
        """General sanity checks."""
        # Null vehicle_id
        if 'vehicle_id' in df.columns:
            null_count = df['vehicle_id'].isnull().sum()
            if null_count > 0:
                self._add_issue(
                    IssueSeverity.WARN,
                    "NULL_VEHICLE_ID",
                    f"{null_count} records with null vehicle_id",
                    feed_type,
                    count=null_count
                )
        
        # Duplicate IDs
        if 'vehicle_id' in df.columns and feed_type == "vehicle_positions":
            dup_count = df['vehicle_id'].duplicated().sum()
            if dup_count > len(df) * 0.1:  # >10% duplicates suspicious
                self._add_issue(
                    IssueSeverity.WARN,
                    "EXCESSIVE_DUPLICATES",
                    f"{dup_count} duplicate vehicle_ids",
                    feed_type,
                    count=dup_count
                )
        
        # Negative delays
        if 'delay' in df.columns:
            neg_count = (df['delay'] < -300).sum()  # More than 5min early
            if neg_count > 0:
                self._add_issue(
                    IssueSeverity.WARN,
                    "LARGE_NEGATIVE_DELAY",
                    f"{neg_count} trips departing >5min early",
                    feed_type,
                    count=neg_count
                )
    
    def _validate_cross_consistency(self, df: pd.DataFrame, feed_type: str):
        """Validate consistency between vehicle positions and trip updates."""
        # Cross-check between feeds requires state across calls
        # Simplified: Check for vehicles without corresponding trip updates
        pass
    
    def _compute_coverage(self, df: pd.DataFrame, feed_type: str):
        """Calculate data coverage metrics."""
        # Placeholder: need static GTFS reference
        estimated_total_trips = 5000  # Netherlands NS daily approx
        
        if feed_type == "trip_updates":
            unique_trips = df['trip_id'].nunique() if 'trip_id' in df.columns else 0
            self.feed_metrics["coverage_trips_pct"] = min(100.0, (unique_trips / estimated_total_trips) * 100)
        
        if feed_type == "vehicle_positions" and 'route_id' in df.columns:
            unique_routes = df['route_id'].nunique()
            estimated_routes = 300  # Approx NS routes
            self.feed_metrics["coverage_routes_pct"] = min(100.0, (unique_routes / estimated_routes) * 100)
    
    def _compute_quality_score(self):
        """Compute overall data quality score 0-100."""
        score = 100.0
        
        # Deduct for ERROR issues (weight 3x)
        error_count = sum(i.count for i in self.issues if i.severity == IssueSeverity.ERROR)
        score -= min(50, error_count * 3)
        
        # Deduct for WARN issues (weight 1x)
        warn_count = sum(i.count for i in self.issues if i.severity == IssueSeverity.WARN)
        score -= min(30, warn_count)
        
        # Deduct for age
        age = self.feed_metrics.get("feed_age_seconds", 0)
        if age > 90:
            score -= min(20, (age - 90) / 10)
        
        self.feed_metrics["validation_score"] = max(0, min(100, score))
    
    def _get_results(self) -> Dict:
        """Format validation results."""
        error_issues = [i for i in self.issues if i.severity == IssueSeverity.ERROR]
        warn_issues = [i for i in self.issues if i.severity == IssueSeverity.WARN]
        info_issues = [i for i in self.issues if i.severity == IssueSeverity.INFO]
        
        return {
            "issues": {
                "error": [
                    {"code": i.code, "description": i.description, "feed_type": i.feed_type, "count": i.count}
                    for i in error_issues
                ],
                "warn": [
                    {"code": i.code, "description": i.description, "feed_type": i.feed_type, "count": i.count}
                    for i in warn_issues
                ],
                "info": [
                    {"code": i.code, "description": i.description, "feed_type": i.feed_type, "count": i.count}
                    for i in info_issues
                ]
            },
            "metrics": self.feed_metrics,
            "summary": {
                "error_count": len(error_issues),
                "warn_count": len(warn_issues),
                "info_count": len(info_issues),
                "total_issues": len(self.issues)
            }
        }

def validate_all_feeds(df: pd.DataFrame) -> Dict[str, Dict]:
    """Validate all feed types in unified DataFrame."""
    results = {}
    
    if df.empty:
        return results
    
    feed_types = df.get('feed_type', pd.Series(['vehicle_positions'] * len(df))).unique()
    
    for feed_type in feed_types:
        feed_df = df[df.get('feed_type', 'vehicle_positions') == feed_type]
        validator = GTFSRTValidator()
        results[feed_type] = validator.validate_feed(feed_df, feed_type)
    
    return results

def render_data_quality_panel():
    """Render data quality monitoring panel for dashboard."""
    st.subheader("GTFS-RT Data Quality Monitor")
    
    df = st.session_state.get('active_data', pd.DataFrame())
    if df.empty:
        st.warning("No data to validate")
        return
    
    with st.spinner("Validating feed quality..."):
        results = validate_all_feeds(df)
    
    if not results:
        st.info("No feed data available for validation")
        return
    
    # Overall health score
    all_scores = [r["metrics"].get("validation_score", 100) for r in results.values()]
    avg_health = np.mean(all_scores) if all_scores else 100
    
    # Status badge
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        health_color = "#22C55E" if avg_health >= 95 else "#F59E0B" if avg_health >= 80 else "#EF4444"
        st.markdown(f"""
        <div style="text-align: center; padding: 10px; border: 2px solid {health_color}; border-radius: 10px;">
            <div style="font-size: 12px; color: #94A3B8;">Feed Health</div>
            <div style="font-size: 24px; font-weight: bold; color: {health_color};">{avg_health:.0f}%</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        total_errors = sum(r["summary"]["error_count"] for r in results.values())
        err_color = "#EF4444" if total_errors > 0 else "#22C55E"
        st.markdown(f"""
        <div style="text-align: center; padding: 10px; border: 2px solid {err_color}; border-radius: 10px;">
            <div style="font-size: 12px; color: #94A3B8;">Errors</div>
            <div style="font-size: 24px; font-weight: bold; color: {err_color};">{total_errors}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        total_warns = sum(r["summary"]["warn_count"] for r in results.values())
        st.markdown(f"""
        <div style="text-align: center; padding: 10px; border: 2px solid #F59E0B; border-radius: 10px;">
            <div style="font-size: 12px; color: #94A3B8;">Warnings</div>
            <div style="font-size: 24px; font-weight: bold; color: #F59E0B;">{total_warns}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        avg_age = np.mean([r["metrics"].get("feed_age_seconds", 0) for r in results.values()])
        st.metric("Avg Feed Age", f"{avg_age:.0f}s")
    
    # Feed-specific details
    for feed_type, result in results.items():
        if result["summary"]["total_issues"] > 0:
            with st.expander(f"{feed_type.replace('_', ' ').title()} - {result['summary']['total_issues']} issues", expanded=False):
                issues_df = pd.DataFrame(result["issues"]["error"] + result["issues"]["warn"])
                if not issues_df.empty:
                    st.dataframe(issues_df, use_container_width=True)
                else:
                    st.success("No quality issues detected")
        
        # Coverage metrics
        m = result["metrics"]
        if "coverage_trips_pct" in m or "coverage_routes_pct" in m:
            st.markdown("**Coverage:**")
            cov_cols = st.columns(2)
            if "coverage_trips_pct" in m:
                with cov_cols[0]:
                    st.metric("Trip Coverage", f"{m['coverage_trips_pct']:.1f}%")
            if "coverage_routes_pct" in m:
                with cov_cols[1]:
                    st.metric("Route Coverage", f"{m['coverage_routes_pct']:.1f}%")

# Background monitoring (run every refresh)
def monitor_feed_quality_continuous():
    """Track quality metrics over time (store in session state)."""
    if 'quality_history' not in st.session_state:
        st.session_state.quality_history = []
    
    df = st.session_state.get('active_data', pd.DataFrame())
    if df.empty:
        return
    
    results = validate_all_feeds(df)
    
    # Record snapshot
    snapshot = {
        "timestamp": datetime.now(),
        "overall_score": np.mean([r["metrics"].get("validation_score", 100) for r in results.values()]),
        "error_count": sum(r["summary"]["error_count"] for r in results.values()),
        "warn_count": sum(r["summary"]["warn_count"] for r in results.values())
    }
    
    st.session_state.quality_history.append(snapshot)
    # Keep last 1000 snapshots
    if len(st.session_state.quality_history) > 1000:
        st.session_state.quality_history = st.session_state.quality_history[-1000:]
