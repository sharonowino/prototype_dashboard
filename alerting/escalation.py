"""
Tiered Alert Escalation Engine
Implements Newark Metro's severity matrix and escalation policies
"""

from datetime import datetime, timedelta
from typing import Dict, List, Optional
import pandas as pd
import streamlit as st

class AlertEscalationEngine:
    """Implements multi-tier alert classification and escalation workflow."""
    
    def __init__(self, config: Dict = None):
        self.config = config or ESCALATION_CONFIG
        self.thresholds = self.config["thresholds"]
        self.tiers = self.config["tiers"]
        
        # Track alert lifecycle
        self.alert_registry: Dict[str, Dict] = {}
    
    def evaluate_prediction(self, prediction: Dict, context: Dict = None) -> Dict:
        """
        Evaluate a disruption prediction and classify alert tier.
        
        Args:
            prediction: Model prediction with severity_class, confidence, delay_minutes
            context: Additional context (route, time, operator, etc.)
        
        Returns:
            Enhanced prediction with tier info, required actions, notification channels
        """
        severity = prediction.get("severity_class", 0)
        confidence = prediction.get("confidence", 100)
        delay_min = prediction.get("delay_minutes", 0)
        bunching = prediction.get("bunching_index", 0)
        
        # Determine base tier
        if severity >= self.thresholds["severity_critical"]:
            tier = "CRITICAL"
        elif delay_min >= self.thresholds["delay_major"]:
            tier = "MAJOR"
        elif delay_min >= self.thresholds["delay_minor"] or bunching >= self.thresholds["bunching_severe"]:
            tier = "MODERATE"
        else:
            tier = "MINOR"
        
        # Get tier policy
        policy = self.tiers[tier]
        
        # Confidence adjustments
        requires_human = policy["requires_human"]
        if confidence < self.thresholds["confidence_escalate_max"]:
            # Low confidence: escalate regardless
            requires_human = True
            tier = "MAJOR" if tier == "MODERATE" else tier
        
        # Auto-publish check
        auto_publish = (
            not requires_human and 
            confidence >= self.thresholds["confidence_auto_min"]
        )
        
        # Build alert object
        alert = {
            **prediction,
            "tier": tier,
            "requires_human_review": requires_human,
            "auto_publish": auto_publish,
            "notification_channels": policy["channels"],
            "response_deadline_minutes": policy["notify_within_minutes"],
            "requires_acknowledgement": policy.get("requires_ack", False),
            "escalation_if_unacked_min": policy.get("escalate_if_unacked"),
            "evaluation_timestamp": datetime.now().isoformat(),
            "status": "NEW"
        }
        
        # Register alert
        alert_id = f"{prediction.get('route_id', 'UNK')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.alert_registry[alert_id] = alert
        
        return alert
    
    def batch_evaluate(self, predictions: List[Dict]) -> List[Dict]:
        """Evaluate multiple predictions."""
        return [self.evaluate_prediction(p) for p in predictions]
    
    def acknowledge_alert(self, alert_id: str, user_id: str) -> bool:
        """Mark alert as acknowledged."""
        if alert_id in self.alert_registry:
            self.alert_registry[alert_id]["status"] = "ACKNOWLEDGED"
            self.alert_registry[alert_id]["acknowledged_by"] = user_id
            self.alert_registry[alert_id]["acknowledged_at"] = datetime.now().isoformat()
            return True
        return False
    
    def resolve_alert(self, alert_id: str, resolution: str) -> bool:
        """Mark alert as resolved."""
        if alert_id in self.alert_registry:
            self.alert_registry[alert_id]["status"] = "RESOLVED"
            self.alert_registry[alert_id]["resolution"] = resolution
            self.alert_registry[alert_id]["resolved_at"] = datetime.now().isoformat()
            return True
        return False
    
    def escalate_overdue_alerts(self) -> List[Dict]:
        """Find alerts past response deadline requiring escalation."""
        overdue = []
        now = datetime.now()
        
        for aid, alert in self.alert_registry.items():
            if alert["status"] not in ["ACKNOWLEDGED", "RESOLVED"]:
                eval_time = datetime.fromisoformat(alert["evaluation_timestamp"])
                deadline = eval_time + timedelta(minutes=alert["response_deadline_minutes"])
                
                if now > deadline:
                    alert["escalated"] = True
                    alert["escalated_at"] = now.isoformat()
                    overdue.append(alert)
        
        return overdue
    
    def get_active_alerts(self, tier_filter: List[str] = None) -> List[Dict]:
        """Get currently active alerts, optionally filtered by tier."""
        alerts = [
            {**alert, "alert_id": aid}
            for aid, alert in self.alert_registry.items()
            if alert["status"] not in ["RESOLVED"]
        ]
        
        if tier_filter:
            alerts = [a for a in alerts if a["tier"] in tier_filter]
        
        return sorted(alerts, key=lambda x: (
            {"CRITICAL": 0, "MAJOR": 1, "MODERATE": 2, "MINOR": 3}[x["tier"]],
            -x.get("severity_class", 0)
        ))

def apply_tiered_escalation(predictions: List[Dict]) -> List[Dict]:
    """Apply tiered escalation to predictions (main entry point)."""
    engine = AlertEscalationEngine()
    return engine.batch_evaluate(predictions)

def render_alert_lifecycle_panel():
    """Display alert lifecycle management UI."""
    st.subheader("Alert Lifecycle Management")
    
    # Get active alerts from session
    if 'alert_engine' not in st.session_state:
        st.session_state.alert_engine = AlertEscalationEngine()
    
    engine = st.session_state.alert_engine
    
    # Overdue alerts
    overdue = engine.escalate_overdue_alerts()
    if overdue:
        st.error(f"⚠️ {len(overdue)} alerts require immediate escalation!")
        for alert in overdue[:5]:
            st.markdown(f"- **{alert.get('route_id')}** ({alert['tier']}) - overdue by "
                       f"{(datetime.now() - datetime.fromisoformat(alert['evaluation_timestamp'])).seconds // 60}min")
    
    # Active alerts table
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("Refresh Alerts", key="refresh_alerts"):
            st.rerun()
    with col2:
        if st.button("Acknowledge All", key="ack_all"):
            for alert in engine.get_active_alerts():
                engine.acknowledge_alert(alert["alert_id"], "current_user")
            st.success("All alerts acknowledged")
    with col3:
        if st.button("Export Log", key="export_alert_log"):
            alerts_df = pd.DataFrame([
                {**a, "alert_id": aid} for aid, a in engine.alert_registry.items()
            ])
            csv = alerts_df.to_csv(index=False)
            st.download_button("Download CSV", csv, "alert_log.csv", "text/csv")
    
    # Active alerts display
    active = engine.get_active_alerts()
    if not active:
        st.info("No active alerts")
        return
    
    # Quick stats
    stats = {tier: sum(1 for a in active if a["tier"] == tier) for tier in ["CRITICAL", "MAJOR", "MODERATE", "MINOR"]}
    st.markdown("**Active Alerts:** " + " | ".join([f"<span style='color:{get_tier_color(t)};font-weight:bold'>{t}: {c}</span>" 
                                                   for t, c in stats.items()]), 
                unsafe_allow_html=True)
    
    # Display as sortable table
    display_df = pd.DataFrame([
        {
            "Route": a.get("route_id", "N/A"),
            "Tier": a["tier"],
            "Severity": a.get("severity", "N/A"),
            "Delay (min)": a.get("delay_minutes", 0),
            "Confidence": f"{a.get('confidence', 0):.0f}%",
            "Channels": ", ".join(a["notification_channels"]),
            "Status": a["status"],
            "Actions": aid[:8]  # Short ID
        }
        for aid, a in engine.alert_registry.items() if a["status"] != "RESOLVED"
    ])
    
    if not display_df.empty:
        st.dataframe(
            display_df,
            use_container_width=True,
            column_config={
                "Tier": st.column_config.TextColumn("Tier"),
                "Actions": st.column_config.TextColumn("Alert ID")
            }
        )

def get_tier_color(tier: str) -> str:
    """Get color for tier."""
    colors = {
        "CRITICAL": "#EF4444",
        "MAJOR": "#F59E0B",
        "MODERATE": "#3B82F6",
        "MINOR": "#10B981"
    }
    return colors.get(tier, "#6B7280")

def get_escalation_summary() -> Dict:
    """Generate summary for KPI panel."""
    if 'alert_engine' not in st.session_state:
        return {"active": 0, "critical": 0, "pending_ack": 0, "overdue": 0}
    
    engine = st.session_state.alert_engine
    active = engine.get_active_alerts()
    
    return {
        "active": len(active),
        "critical": sum(1 for a in active if a["tier"] == "CRITICAL"),
        "pending_ack": sum(1 for a in active if a["status"] == "NEW" and a.get("requires_ack")),
        "overdue": len(engine.escalate_overdue_alerts())
    }
