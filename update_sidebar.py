import re

with open('dashboard.py', 'r') as f:
    content = f.read()

old = r'(    # Sidebar Navigation.*?\n)(    # Header.*?\n    render_header\(\))'
new = '''    # Sidebar Navigation
    st.sidebar.title("Navigation")
    pages = ["Overview", "Live Map", "Alerts", "Predictions", "Analytics", "Settings"]
    selection = st.sidebar.radio("Go to", pages, key="nav_radio_main")
    st.session_state.page = selection

    # System Overview
    st.sidebar.markdown("---")
    st.sidebar.markdown("### System Overview")
    df_sidebar = st.session_state.active_data if not st.session_state.active_data.empty else gen_demo_data(20)
    preds_sidebar = st.session_state.predictions if st.session_state.predictions else run_predictions(df_sidebar)
    total_routes = len(df_sidebar)
    normal_cnt = sum(1 for p in preds_sidebar if p["severity_class"] == 0)
    disrupted_cnt = len(preds_sidebar) - normal_cnt

    # KPIs
    st.sidebar.markdown("### KPIs")
    kpi_col1, kpi_col2 = st.sidebar.columns(2)
    with kpi_col1:
        st.sidebar.metric("Total Routes", total_routes)
        st.sidebar.metric("Normal", normal_cnt, delta_color="normal")
    with kpi_col2:
        st.sidebar.metric("Disrupted", disrupted_cnt, delta_color="inverse")
        st.sidebar.metric("On-Time Perf", "88.2%", delta="-1.8%")

    st.sidebar.markdown("---")

    # Traffic Mgmt
    st.sidebar.markdown("### Traffic Mgmt")
    if not df_sidebar.empty and "delay_min" in df_sidebar.columns:
        incident_resp = df_sidebar["delay_min"].mean()
    else:
        incident_resp = 15.0
    tm_col1, tm_col2 = st.sidebar.columns(2)
    with tm_col1:
        st.sidebar.metric("Incident Resp", f"{incident_resp:.0f} min")
        st.sidebar.metric("T-Time Idx", "1.30")
    with tm_col2:
        st.sidebar.metric("Congestion", "2.5 min")
        st.sidebar.metric("Route Eff", "85.0%")

    # Active Alerts
    st.sidebar.markdown("---")
    st.sidebar.markdown("### Active Alerts")
    active_alerts = [p for p in preds_sidebar if p["severity_class"] > 0]
    if active_alerts:
        for alert in active_alerts[:3]:
            sev = SEVERITY_CONFIG[alert["severity_class"]]
            st.sidebar.markdown(f\"<div style=\"background:{sev[\"color\"]}22;padding:4px;border-radius:3px;margin:2px 0;border-left:2px solid {sev[\"color\"]}\"><div style=\"font-size:10px;font-weight:bold;color:{sev[\"color\"]}\">{sev[\"label\"]}</div><div style=\"font-size:9px\">{alert[\"route_id\"]} ({alert[\"confidence\"]:.0f}%)</div></div>\", unsafe_allow_html=True)
    else:
        st.sidebar.markdown(\"No active alerts\")

    st.sidebar.markdown(\"---\")

    # Data Source
    st.sidebar.markdown(\"**Data Source**\")
    data_source = st.sidebar.radio(\"\", [\"Demo Data\", \"Parquet File\", \"Upload ZIP\"], key=\"ds_settings\", label_visibility=\"collapsed\")
    st.session_state.data_source_type = data_source
    if data_source == \"Parquet File\":
        path = st.sidebar.text_input(\"Path\", \"merged_with_alerts.parquet\", key=\"parquet_path\")
    elif data_source == \"Upload ZIP\":
        uploaded = st.sidebar.file_uploader(\"ZIP\", type=[\"zip\"], key=\"zip_upload\")
        path = uploaded if uploaded else None
    else:
        path = None

    # Model Config
    st.sidebar.markdown(\"**Model Config**\")
    st.sidebar.selectbox(\"Model\", [\"RandomForest\", \"XGBoost\", \"NeuralNet\", \"BEST\"], key=\"model_type_select\", label_visibility=\"collapsed\")
    st.sidebar.slider(\"Confidence\", 0.0, 1.0, 0.5, 0.05, key=\"confidence_slider\", label_visibility=\"collapsed\")

    # Display
    st.sidebar.markdown(\"**Display**\")
    st.sidebar.checkbox(\"Auto Refresh\", False, key=\"auto_refresh_chk\")
    st.sidebar.slider(\"Interval (s)\", 10, 300, 60, 10, key=\"refresh_interval_slider\", label_visibility=\"collapsed\")
    st.sidebar.checkbox(\"Show Predictions\", True, key=\"show_predictions_chk\")

    st.sidebar.markdown(\"---\")

    # Refresh
    if st.sidebar.button(\"Refresh\", key=\"refresh_btn\", use_container_width=True):
        st.session_state.active_data = pd.DataFrame()
        st.session_state.predictions = []
        st.rerun()

    # Load data
    if st.session_state.active_data.empty:
        if data_source == \"Demo Data\":
            st.session_state.active_data = gen_demo_data(20)
            st.session_state.predictions = run_predictions(st.session_state.active_data)
        elif data_source == \"Upload ZIP\" and path is not None:
            with st.spinner(\"Processing...\"):
                df = extract_zip_upload(path)
                if not df.empty:
                    st.session_state.active_data = df
                    st.session_state.predictions = run_predictions(df)
                else:
                    st.session_state.active_data = gen_demo_data(20)
                    st.session_state.predictions = run_predictions(st.session_state.active_data)
        elif data_source == \"Parquet File\":
            with st.spinner(\"Loading...\"):
                df = load_from_parquet(path)
                if not df.empty:
                    st.session_state.active_data = df
                    st.session_state.predictions = run_predictions(df)
                else:
                    st.session_state.active_data = gen_demo_data(20)
                    st.session_state.predictions = run_predictions(st.session_state.active_data)
        else:
            st.session_state.active_data = load_from_parquet(\"merged_with_alerts.parquet\")
            if st.session_state.active_data.empty:
                st.session_state.active_data = gen_demo_data(20)
            st.session_state.predictions = run_predictions(st.session_state.active_data)

    # Main Content
    render_header()
'''

result = re.sub(old, new, content, flags=re.DOTALL)
if result != content:
    with open('dashboard.py', 'w') as f:
        f.write(result)
    print('Replaced successfully')
else:
    print('No change - regex did not match')
