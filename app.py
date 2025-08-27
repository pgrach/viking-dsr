import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import yaml
from datetime import datetime, timedelta
import os, re, glob
from pathlib import Path
from hydro_miner_model import (
    load_hydro_data, load_btc_data, analyze_power_profile,
    run_monte_carlo_simulation, calculate_optimal_fleet,
    project_mining_economics
)

# Page configuration
st.set_page_config(
    page_title="Power-Constrained Bitcoin Mining Optimizer",
    page_icon="⚡",
    layout="wide"
)

# Load configuration
@st.cache_data
def load_config():
    with open('config.yaml', 'r') as file:
        return yaml.safe_load(file)

# Helper: find latest Viking CSV by end date in filename, fallback to modified time
def find_latest_viking_csv(search_dir: str = "."):
    pattern = os.path.join(search_dir, "viking_halfhourly_curtailment_*.csv")
    files = glob.glob(pattern)
    if not files:
        return None
    rx = re.compile(r"viking_halfhourly_curtailment_(\d{4}-\d{2}-\d{2})_to_(\d{4}-\d{2}-\d{2})\.csv$", re.IGNORECASE)
    best = None
    for f in files:
        m = rx.search(os.path.basename(f))
        if m:
            try:
                end_date = datetime.strptime(m.group(2), "%Y-%m-%d").date()
            except ValueError:
                end_date = None
            if end_date is not None and (best is None or end_date > best[0]):
                best = (end_date, f)
    if best:
        return best[1]
    # Fallback to newest by modified time
    return max(files, key=os.path.getmtime)

# Load data functions
@st.cache_data
def get_hydro_analysis(hydro_csv_path: str, file_mtime: float):
    # file_mtime included to invalidate cache when source file updates
    hydro_data = load_hydro_data(hydro_csv_path)
    stats = analyze_power_profile(hydro_data)
    # Attach metadata for transparency
    stats['source_file'] = os.path.basename(hydro_csv_path)
    try:
        stats['n_rows'] = len(hydro_data)
        stats['start_ts'] = pd.to_datetime(hydro_data['timestamp']).min()
        stats['end_ts'] = pd.to_datetime(hydro_data['timestamp']).max()
    except Exception:
        stats['n_rows'] = None
    return stats

@st.cache_data
def get_btc_data():
    return load_btc_data('btc_price.csv', 'btc_difficulty.csv')

# Clear cache if needed (uncomment the next line and reload if you see cache-related errors)
# get_btc_data.clear()

# Initialize session state
if 'simulation_results' not in st.session_state:
    st.session_state.simulation_results = None

# Header
st.title("⚡ Power-Constrained Bitcoin Mining Optimizer")
st.markdown("*Optimize ASIC fleet size for variable, intermittent power (hydro, wind, solar, grid-flex, etc.)*")

# Sidebar configuration
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # Load default config and data
    config = load_config()
    btc_data = get_btc_data()
    
    # Hydro data source detection
    latest_hydro_csv = find_latest_viking_csv(".")
    if latest_hydro_csv is None:
        st.error("No viking_halfhourly_curtailment_*.csv found in the project directory.")
        st.stop()
    hydro_file_mtime = os.path.getmtime(latest_hydro_csv)
    st.session_state['hydro_csv_path'] = latest_hydro_csv
    st.session_state['hydro_csv_mtime'] = hydro_file_mtime

    st.caption(f"📄 Using power CSV: {Path(latest_hydro_csv).name}")
    if st.button("Refresh hydro data cache"):
        get_hydro_analysis.clear()
        st.rerun()
    
    # Clear cache if old format is detected (missing data_date)
    if 'data_date' not in btc_data:
        st.info("🔄 Updating data format... Please refresh the page.")
        get_btc_data.clear()
        st.stop()
    
    # Quick Start Guide
    with st.expander("🚀 Quick Start Guide", expanded=False):
        st.markdown("""
        **Follow these steps:**
        1. **Configure your ASIC model** below
        2. **Set financial parameters** (costs, fees)
        3. **Adjust economic parameters** as needed (defaults based on historical data)
        4. **Click 'Run Optimization'** to get results
        
        💡 **Tip**: Start with default values, then adjust based on your specific situation.
        """)
    
    st.subheader("🔧 ASIC Specifications")
    asic_model = st.text_input(
        "Model", 
        value=config['asic']['model'],
        help="ASIC miner model name for reference"
    )
    
    # Overclocking capability toggle
    enable_overclocking = st.checkbox(
        "🚀 Enable Overclocking Mode",
        value=config.get('asic', {}).get('overclocking_enabled', True),
        help="Allow ASICs to run in overclocked mode when excess power is available"
    )
    
    if enable_overclocking:
        st.info("💡 **Overclocking Mode**: ASICs will dynamically switch between standard and overclocked modes based on available power")
        
        col_base, col_oc = st.columns(2)
        with col_base:
            st.markdown("**Standard Mode**")
            asic_hashrate = st.number_input(
                "Base Hashrate (TH/s)", 
                value=config['asic']['hash_rate_th'], 
                min_value=1,
                help="Standard mining speed in terahashes per second"
            )
            asic_power = st.number_input(
                "Base Power per TH (W)", 
                value=config['asic']['watts_per_th'], 
                min_value=1.0,
                help="Standard power consumption per terahash"
            )
        
        with col_oc:
            st.markdown("**Overclocked Mode**")
            asic_hashrate_oc = st.number_input(
                "OC Hashrate (TH/s)", 
                value=562.5,  # Your overclocked specs
                min_value=float(asic_hashrate),
                help="Overclocked mining speed in terahashes per second"
            )
            asic_power_oc = st.number_input(
                "OC Power per TH (W)", 
                value=18.7,  # Your overclocked specs
                min_value=1.0,
                help="Overclocked power consumption per terahash"
            )
    else:
        col_hash, col_power = st.columns(2)
        with col_hash:
            asic_hashrate = st.number_input(
                "Hashrate (TH/s)", 
                value=config['asic']['hash_rate_th'], 
                min_value=1,
                help="Mining speed in terahashes per second"
            )
        with col_power:
            asic_power = st.number_input(
                "Power per TH (W)", 
                value=config['asic']['watts_per_th'], 
                min_value=1.0,
                help="Power consumption per terahash"
            )
        
        # Set overclocked values same as standard when overclocking is disabled
        asic_hashrate_oc = asic_hashrate
        asic_power_oc = asic_power
    
    asic_price = st.number_input(
        "Price per TH ($)", 
        value=config['asic']['price_usd_per_th'], 
        min_value=1.0,
        help="Cost per terahash of mining equipment"
    )
    
    # Show calculated metrics
    total_asic_power = asic_hashrate * asic_power / 1000
    total_asic_cost = asic_hashrate * asic_price
    
    if enable_overclocking:
        total_asic_power_oc = asic_hashrate_oc * asic_power_oc / 1000
        efficiency_standard = asic_hashrate / total_asic_power  # TH/kW
        efficiency_oc = asic_hashrate_oc / total_asic_power_oc  # TH/kW
        efficiency_gain = ((efficiency_oc - efficiency_standard) / efficiency_standard) * 100
        
        st.info(f"""
        📊 **Per ASIC Comparison**:
        - **Standard**: {total_asic_power:.1f} kW, {asic_hashrate} TH/s ({efficiency_standard:.1f} TH/kW)
        - **Overclocked**: {total_asic_power_oc:.1f} kW, {asic_hashrate_oc} TH/s ({efficiency_oc:.1f} TH/kW)
        - **Cost**: ${total_asic_cost:,.0f} (same for both modes)
        - **Efficiency change**: {efficiency_gain:+.1f}%
        """)
    else:
        st.info(f"📊 **Per ASIC**: {total_asic_power:.1f} kW power, ${total_asic_cost:,.0f} cost")
    
    st.subheader("💰 Financial Assumptions")
    
    col_pool, col_discount = st.columns(2)
    with col_pool:
        pool_fee = st.number_input(
            "Pool Fee (%)",
            value=config['financial']['pool_fee_percent'] * 100,
            min_value=0.0,
            max_value=10.0,
            step=0.1,
            format="%.1f",
            help="Mining pool fee (typically 1-2%)"
        ) / 100.0
    
    with col_discount:
        discount_rate = st.number_input(
            "Discount Rate (%)",
            value=float(config.get('financial', {}).get('discount_rate_percent', 0.10) * 100),
            min_value=0.0,
            max_value=50.0,
            step=0.5,
            format="%.1f",
            help="Required rate of return for NPV calculations"
        ) / 100.0

    st.subheader("🏭 Operating Costs")
    annual_opex = st.number_input(
        "Annual Operating Costs ($)", 
        value=config['operating_costs']['annual_usd'], 
        min_value=0,
        help="Yearly costs: maintenance, cooling, monitoring, etc."
    )
    additional_upfront_costs = st.number_input(
        "Additional Upfront Costs ($)", 
        value=int(config.get('financial', {}).get('additional_upfront_costs', 0)), 
        min_value=0,
        help="Setup costs: installation, infrastructure, permits, etc."
    )
    
    st.subheader("🎲 Simulation Parameters")
    
    col_sim, col_step = st.columns(2)
    with col_sim:
        n_simulations = st.number_input(
            "Monte Carlo Runs", 
            value=int(config['simulation']['n_simulations']), 
            min_value=100, 
            max_value=1000,
            step=100,
            help="Number of simulation runs (fewer = faster, recommended 500)"
        )
    with col_step:
        fleet_step = st.number_input(
            "Fleet Size Step", 
            value=20, 
            min_value=5, 
            max_value=100,
            step=5,
            help="ASIC increment for testing different fleet sizes (larger = faster, recommended 20+)"
        )
    
    projection_years = st.number_input(
        "Projection Years", 
        value=config['simulation']['projection_years'], 
        min_value=1, 
        max_value=5,
        help="Investment time horizon for analysis"
    )
    
    st.subheader("📈 Economic Parameters")
    
    # Use only the custom scenario; defaults loaded from config
    selected_scenario = config['scenarios']['custom'].copy()
    selected_scenario['difficulty_growth_annual'] = float(selected_scenario.get('difficulty_growth_annual', 0.10))
    selected_scenario['price_change_annual'] = float(selected_scenario.get('price_change_annual', 0.10))
    selected_scenario['price_volatility_annual'] = float(selected_scenario.get('price_volatility_annual', 0.10))
    
    st.markdown("**🎛️ Adjust Parameters (defaults loaded from configuration)**")

    # Helper function to round to the nearest step for sliders
    def round_to_step(value, step):
        return step * round(value / step)
    
    col_diff, col_price = st.columns(2)
    with col_diff:
        difficulty_default = np.clip(np.nan_to_num(selected_scenario['difficulty_growth_annual'] * 100), -50.0, 150.0)
        custom_difficulty_growth = st.slider(
            "Difficulty Growth (%/year)",
            min_value=-50.0,
            max_value=150.0,
            value=round_to_step(difficulty_default, 1.0),
            step=1.0,
            format="%.1f%%",
            help="Expected annual change in mining difficulty"
        ) / 100.0
    
    with col_price:
        price_default = np.clip(np.nan_to_num(selected_scenario['price_change_annual'] * 100), -50.0, 150.0)
        custom_price_change = st.slider(
            "Price Change (%/year)",
            min_value=-50.0,
            max_value=150.0,
            value=round_to_step(price_default, 1.0),
            step=1.0,
            format="%.1f%%",
            help="Expected annual Bitcoin price growth"
        ) / 100.0
    
    volatility_default = np.clip(np.nan_to_num(selected_scenario['price_volatility_annual'] * 100), 0.0, 200.0)
    custom_price_volatility = st.slider(
        "Price Volatility (%/year)",
        min_value=0.0,
        max_value=200.0,
        value=round_to_step(volatility_default, 5.0),
        step=5.0,
        format="%.1f%%",
        help="Expected annual price volatility (uncertainty)"
    ) / 100.0
    
    selected_scenario['difficulty_growth_annual'] = custom_difficulty_growth
    selected_scenario['price_change_annual'] = custom_price_change
    selected_scenario['price_volatility_annual'] = custom_price_volatility

    # (Reverted) Projections will use median of annual average price as before

    with st.expander("📋 Current Parameters", expanded=True):
        st.markdown(f"**{selected_scenario['name']}**")
        st.markdown(f"*{selected_scenario['description']}*")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric(
                "Difficulty Growth", 
                f"{selected_scenario['difficulty_growth_annual']:.1%}",
                help="Expected annual change in Bitcoin mining difficulty"
            )
        with col2:
            st.metric(
                "Price Growth", 
                f"{selected_scenario['price_change_annual']:.1%}",
                help="Expected annual change in Bitcoin price"
            )
        with col3:
            st.metric(
                "Volatility", 
                f"{selected_scenario['price_volatility_annual']:.1%}",
                help="Higher volatility = more uncertainty in projections"
            )

    # Enhanced run button with validation
    st.markdown("---")
    
    run_simulation = st.button(
        "🚀 Run Optimization", 
        type="primary",
        help="Start Monte Carlo simulation to find optimal fleet size",
        use_container_width=True
    )

# Main content area
col1, col2 = st.columns([3, 2])

with col1:
    st.header("📊 Power Analysis")
    hydro_stats = get_hydro_analysis(
        st.session_state['hydro_csv_path'],
        st.session_state['hydro_csv_mtime']
    )
    st.caption(f"📂 Power data source: {hydro_stats.get('source_file','unknown')} • Intervals: {hydro_stats.get('n_rows','?')}")
    
    # Add validation warning if power is very low
    if hydro_stats['max_power_kw'] < 50:
        st.warning("⚠️ **Low Power Alert**: Your site has very low available power. Consider reviewing the data or consulting with a power systems engineer.")
    
    # Display power statistics
    metrics_col1, metrics_col2, metrics_col3, metrics_col4 = st.columns(4)
    with metrics_col1:
        st.metric(
            "🔋 Average Power", 
            f"{hydro_stats['avg_power_kw']:.0f} kW", 
            help="Average power output including all downtime periods"
        )
    with metrics_col2:
        st.metric(
            "⏰ Uptime", 
            f"{hydro_stats['uptime_percent']:.1f}%", 
            help="Percentage of time the site generates any power"
        )
    with metrics_col3:
        st.metric(
            "🎯 Baseload (When ON)", 
            f"{hydro_stats['op_p10_power_kw']:.0f} kW",
            help="Power level exceeded 90% of the time when plant is operational (excludes zero-power periods)"
        )
    with metrics_col4:
        st.metric(
            "📈 Peak Power", 
            f"{hydro_stats['max_power_kw']:.0f} kW",
            help="Maximum recorded power output"
        )
    

    # Power capacity insights
    asic_power_kw = asic_hashrate * asic_power / 1000
    max_asics = int(hydro_stats['max_power_kw'] / asic_power_kw)
    baseload_asics = int(hydro_stats['op_p10_power_kw'] / asic_power_kw)

    if enable_overclocking:
        asic_power_kw_oc = asic_hashrate_oc * asic_power_oc / 1000
        max_asics_oc = int(hydro_stats['max_power_kw'] / asic_power_kw_oc)
        baseload_asics_oc = int(hydro_stats['op_p10_power_kw'] / asic_power_kw_oc)
        st.info(f"""
        💡 **Power Insights for {asic_model} with Overclocking**:
        - **Standard mode capacity**: {max_asics} ASICs ({max_asics * asic_power_kw:.0f} kW)
        - **Overclocked mode capacity**: {max_asics_oc} ASICs ({max_asics_oc * asic_power_kw_oc:.0f} kW)
        - **Reliable baseload (standard)**: {baseload_asics} ASICs can run 90% of the time when plant is ON (effective {baseload_asics * asic_power_kw / hydro_stats['max_power_kw'] * 100:.0f}% of max capacity)
        - **Reliable baseload (overclocked)**: {baseload_asics_oc} ASICs can run 90% of the time when plant is ON (effective {baseload_asics_oc * asic_power_kw_oc / hydro_stats['max_power_kw'] * 100:.0f}% of max capacity)
        - **Smart scaling**: Fleet automatically adjusts between modes based on available power
    - **Note**: Site uptime is {hydro_stats['uptime_percent']:.1f}%, so actual availability varies
        """)
    else:
        st.info(f"""
        💡 **Power Insights for {asic_model}**:
        - **Maximum capacity**: {max_asics} ASICs ({max_asics * asic_power_kw:.0f} kW)
        - **Reliable baseload**: {baseload_asics} ASICs can run 90% of the time when plant is ON (effective {baseload_asics * asic_power_kw / hydro_stats['max_power_kw'] * 100:.0f}% of max capacity)
        - **Power per ASIC**: {asic_power_kw:.1f} kW
    - **Note**: Site uptime is {hydro_stats['uptime_percent']:.1f}%, so actual availability varies
        """)

    # --- Fleet sizes to test: always define for plotting and simulation ---
    max_possible_asics = max_asics
    num_fleet_sizes = min(20, max_possible_asics)
    if num_fleet_sizes > 1:
        fleet_sizes_to_test = np.linspace(1, max_possible_asics, num=num_fleet_sizes, dtype=int)
        fleet_sizes_to_test = np.unique(fleet_sizes_to_test).tolist()
    elif max_possible_asics >= 1:
        fleet_sizes_to_test = [1]
    else:
        fleet_sizes_to_test = []

    # Power duration curve with enhanced visualization
    st.subheader("Power Duration Curve")
    fig_duration = go.Figure()
    percentiles = list(hydro_stats['power_percentiles'].keys())
    power_values = list(hydro_stats['power_percentiles'].values())
    # Main power curve
    fig_duration.add_trace(go.Scatter(
        x=percentiles,
        y=power_values,
        mode='lines',
        name='Available Power',
        fill='tozeroy',
        line=dict(color='#1f77b4', width=3),
        hovertemplate='<b>Power Duration</b><br>Exceedance: %{x}%<br>Power: %{y:.0f} kW<extra></extra>'
    ))
    # Show horizontal lines for only the tested fleet sizes
    colors = ['#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    for i, n_asics in enumerate(fleet_sizes_to_test):
        fig_duration.add_hline(
            y=n_asics * asic_power_kw,
            line_dash="dash",
            line_color=colors[i % len(colors)],
            annotation_text=f"{n_asics} ASICs ({n_asics * asic_power_kw:.0f} kW)",
            annotation_position="right",
            annotation=dict(font_size=10)
        )
    
    fig_duration.update_layout(
        xaxis_title="Exceedance Probability (%)",
        yaxis_title="Power (kW)",
        height=450,
        margin=dict(r=120),
        showlegend=True,
        hovermode='x unified'
    )
    st.plotly_chart(fig_duration, use_container_width=True)
    
    st.info("💡 **How to read this chart**: The curve shows what power level is exceeded for each percentage of time. Horizontal lines show different ASIC fleet power requirements.")
    
    # Monthly variation with enhanced styling
    st.subheader("Monthly Power Variation")
    monthly_data = pd.DataFrame({
        'Month': ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'],
        'Average Power (kW)': hydro_stats['monthly_avg_power']
    })
    
    # Enhanced monthly chart
    fig_monthly = px.bar(
        monthly_data, 
        x='Month', 
        y='Average Power (kW)',
        color='Average Power (kW)', 
        color_continuous_scale='Viridis_r',
    )
    
    fig_monthly.update_traces(
        hovertemplate='<b>%{x}</b><br>Average Power: %{y:.0f} kW<extra></extra>'
    )
    
    # Add average line
    avg_power = np.mean(hydro_stats['monthly_avg_power'])
    fig_monthly.add_hline(
        y=avg_power, 
        line_dash="dash", 
        line_color="red",
        annotation_text=f"Annual Average ({avg_power:.0f} kW)"
    )
    
    fig_monthly.update_layout(
        height=350,
        showlegend=False,
        yaxis_title="Average Power (kW)",
        xaxis_title="Month"
    )
    st.plotly_chart(fig_monthly, use_container_width=True)
    
    # Add seasonal insights
    max_month_idx = np.argmax(hydro_stats['monthly_avg_power'])
    min_month_idx = np.argmin(hydro_stats['monthly_avg_power'])
    month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    
    col_season1, col_season2 = st.columns(2)
    with col_season1:
        st.metric(
            "🌊 Peak Month", 
            month_names[max_month_idx],
            f"{hydro_stats['monthly_avg_power'][max_month_idx]:.0f} kW"
        )
    with col_season2:
        st.metric(
            "🏜️ Low Month", 
            month_names[min_month_idx],
            f"{hydro_stats['monthly_avg_power'][min_month_idx]:.0f} kW"
        )

with col2:
    st.header("💰 Mining Economics")
    
    current_btc_price = btc_data['current_price']
    current_difficulty = btc_data['current_difficulty']
    current_revenue_per_th = btc_data['current_revenue_per_th']
    data_date = btc_data.get('data_date')  # Use .get() for fallback
    
    # Add actual data date info
    if data_date:
        st.caption(f"📅 Data as of: {data_date.strftime('%B %d, %Y')} (latest available)")
    else:
        st.caption(f"📅 Data as of: {datetime.now().strftime('%B %d, %Y')} (cached data)")
    
    metrics_col1, metrics_col2 = st.columns(2)
    with metrics_col1:
        st.metric(
            "🪙 BTC Price", 
            f"${current_btc_price:,.0f}",
            help="Current Bitcoin price used for revenue calculations"
        )
        st.metric(
            "⚡ Revenue/TH/day", 
            f"${current_revenue_per_th:.4f}",
            help="Daily revenue per terahash at current price and difficulty (assuming 100% utilization)"
        )
    with metrics_col2:
        st.metric(
            "🔧 Network Difficulty", 
            f"{current_difficulty/1e12:.1f}T",
            help="Current Bitcoin network mining difficulty (higher = more competition)"
        )
        st.metric(
            "💵 Revenue/ASIC/day", 
            f"${current_revenue_per_th * asic_hashrate:.2f}",
            help=f"Expected daily revenue for your {asic_model} ASIC model (assuming 100% utilization)"
        )
    
    # Historical trends
    st.subheader("Market Trends")
    
    # Create subplot with price and difficulty
    fig_trends = go.Figure()
    
    # Add BTC price on primary y-axis
    fig_trends.add_trace(go.Scatter(
        x=btc_data['historical_data']['date'],
        y=btc_data['historical_data']['price'],
        name='BTC Price',
        yaxis='y',
        line=dict(color='#f7931e', width=2),
        hovertemplate='<b>BTC Price</b><br>Date: %{x}<br>Price: $%{y:,.0f}<extra></extra>'
    ))
    
    # Add difficulty on secondary y-axis
    fig_trends.add_trace(go.Scatter(
        x=btc_data['historical_data']['date'],
        y=btc_data['historical_data']['difficulty'],
        name='Network Difficulty',
        yaxis='y2',
        line=dict(color='#ff4b4b', width=2),
        hovertemplate='<b>Network Difficulty</b><br>Date: %{x}<br>Difficulty: %{y:.2e}<extra></extra>'
    ))
    
    fig_trends.update_layout(
        yaxis=dict(
            title='BTC Price ($)', 
            side='left',
            title_font=dict(color='#f7931e'),
            tickfont=dict(color='#f7931e')
        ),
        yaxis2=dict(
            title='Network Difficulty', 
            side='right', 
            overlaying='y',
            title_font=dict(color='#ff4b4b'),
            tickfont=dict(color='#ff4b4b')
        ),
        height=350,
        hovermode='x unified',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        margin=dict(t=50, b=50, l=50, r=50)
    )
    st.plotly_chart(fig_trends, use_container_width=True)
    
    # Add historical growth metrics based on last 365 days
    st.subheader("Historical 365-Day Performance")
    st.caption("Use these metrics as guidance when setting your scenario parameters")
    
    hist_col1, hist_col2, hist_col3 = st.columns(3)
    with hist_col1:
        st.metric(
            "⚙️ Difficulty Growth",
            f"{btc_data.get('difficulty_growth_annual', 0):.1%}",
            help="Annualized network difficulty change over the last 365 days"
        )
    with hist_col2:
        st.metric(
            "📈 Price Growth",
            f"{btc_data.get('price_growth_annual', 0):.1%}",
            help="Annualized Bitcoin price change over the last 365 days"
        )
    with hist_col3:
        st.metric(
            "📊 Price Volatility",
            f"{btc_data.get('price_volatility_annual', 0):.1%}",
            help="Annualized price volatility over the last 365 days"
        )
    
    # Add comparison with scenario parameters
    if 'selected_scenario' in locals():
        st.info(f"""
        💡 **Scenario Comparison**: Your selected "{selected_scenario['name']}" scenario assumes:
        - Price growth: {selected_scenario['price_change_annual']:.1%} vs historical {btc_data.get('price_growth_annual', 0):.1%}
        - Difficulty growth: {selected_scenario['difficulty_growth_annual']:.1%} vs historical {btc_data.get('difficulty_growth_annual', 0):.1%}
        """)
    
    # Add explanation
    st.info("💡 **Chart Tip**: This shows how Bitcoin price and mining difficulty have evolved over time. Both metrics directly impact mining profitability.")

# Simulation results section

if run_simulation:
    st.header("🎯 Optimization Results")

    # Show estimated runtime
    estimated_time = (n_simulations * len(fleet_sizes_to_test)) / 50000
    if estimated_time < 1:
        time_text = f"~{estimated_time*60:.0f} seconds"
    else:
        time_text = f"~{estimated_time:.1f} minutes"
    st.caption(f"⏱️ Estimated runtime: {time_text}")

    if len(fleet_sizes_to_test) < 1:
        st.error("❌ **Power Insufficient**: Your site cannot power even a single ASIC. Consider using more efficient miners or verify your power data.")
        st.stop()

    # Enhanced progress display
    progress_bar = st.progress(0)
    status_text = st.empty()

    try:
        with st.spinner("🔄 Initializing Monte Carlo simulation..."):
            status_text.text(f"🚀 Starting {n_simulations:,} simulations across {len(fleet_sizes_to_test)} fleet sizes...")
            progress_bar.progress(0.1)

            # Prepare ASIC specs
            asic_specs = {
                'model': asic_model,
                'hash_rate_th': asic_hashrate,
                'watts_per_th': asic_power,
                'price_usd_per_th': asic_price,
                'power_consumption_kw': asic_hashrate * asic_power / 1000,
                'unit_price': asic_hashrate * asic_price,
                'enable_overclocking': enable_overclocking,
                'hash_rate_th_oc': asic_hashrate_oc,
                'watts_per_th_oc': asic_power_oc,
                'power_consumption_kw_oc': asic_hashrate_oc * asic_power_oc / 1000
            }

            progress_bar.progress(0.2)
            status_text.text("⚙️ Running simulations... This may take a few minutes.")

            # Run simulation using only the selected fleet sizes
            simulation_results = run_monte_carlo_simulation(
                hydro_stats=hydro_stats,
                btc_data=btc_data,
                asic_specs=asic_specs,
                annual_opex=annual_opex,
                n_simulations=n_simulations,
                fleet_sizes=fleet_sizes_to_test,
                scenario_params=selected_scenario,
                projection_years=projection_years,
                pool_fee=pool_fee,
                discount_rate=discount_rate,
                additional_upfront_costs=additional_upfront_costs
            )

            progress_bar.progress(0.9)
            status_text.text("📊 Processing results...")

            st.session_state.simulation_results = simulation_results

            progress_bar.progress(1.0)
            status_text.text("✅ Simulation complete!")
            
    except Exception as e:
        st.error(f"❌ **Simulation Error**: {str(e)}")
        st.info("💡 Try reducing the number of simulations or fleet step size, or check your input parameters.")
        st.stop()
    
    finally:
        # Clean up progress indicators
        progress_bar.empty()
        status_text.empty()

    # Display results
    results = st.session_state.simulation_results

    if not results or not results.get('fleet_sizes'):
        st.error("❌ No simulation results generated. Please check your configuration and try again.")
        st.stop()

    # Success message
    st.success(f"🎉 **Optimization Complete!** Analyzed {len(results['fleet_sizes'])} fleet configurations using {n_simulations:,} Monte Carlo simulations.")

    # Optimal fleet recommendation with enhanced display
    st.subheader("🏆 Optimal Fleet Recommendation")

    optimal = calculate_optimal_fleet(results, hydro_stats, asic_specs)

    if optimal and optimal.get('n_asics', 0) > 0:
        st.info(optimal['recommendation'])

        # Display key metrics for the optimal fleet
        opt_col1, opt_col2, opt_col3, opt_col4 = st.columns(4)
        with opt_col1:
            st.metric("Recommended ASICs", f"{optimal['n_asics']}", help="Optimal number of ASICs based on maximizing risk-adjusted returns.")
        with opt_col2:
            st.metric("Expected NPV", f"${optimal['expected_npv']:,.0f}", help="Average Net Present Value over all simulations.")
        with opt_col3:
            st.metric("Payback Period", f"{optimal['payback_months']:.1f} months", help="Median time to recover initial investment.")
        with opt_col4:
            st.metric("IRR (Median)", f"{optimal['irr_median']:.1f}%", help="Median Internal Rate of Return from simulations.")

        # Installed power metrics for the recommended fleet
        installed_power_standard_kw = optimal['n_asics'] * asic_specs['power_consumption_kw']
        if enable_overclocking:
            installed_power_oc_kw = optimal['n_asics'] * asic_specs['power_consumption_kw_oc']
            ip_col1, ip_col2 = st.columns(2)
            with ip_col1:
                st.metric("Installed Power (Standard)", f"{installed_power_standard_kw:,.0f} kW")
            with ip_col2:
                st.metric("Installed Power (OC)", f"{installed_power_oc_kw:,.0f} kW")
        else:
            st.metric("Installed Power", f"{installed_power_standard_kw:,.0f} kW")

        # Utilization (capacity factor) and power-duration statements
        # Find index for the recommended fleet size to pull capacity factor
        try:
            rec_idx = results['fleet_sizes'].index(optimal['n_asics'])
        except ValueError:
            # Fallback to closest size
            fleets = results.get('fleet_sizes', []) or []
            if fleets:
                rec_idx = int(np.argmin([abs(s - optimal['n_asics']) for s in fleets]))
            else:
                rec_idx = 0

        cap_factor = None
        if 'capacity_factor' in results and len(results['capacity_factor']) > rec_idx:
            cap_factor = results['capacity_factor'][rec_idx]

        # Compute H% of hours where available power >= installed power (simulated P50)
        h_standard = None
        h_oc = None
        if 'full_power_percent_std_p50' in results and len(results['full_power_percent_std_p50']) > rec_idx:
            h_standard = float(results['full_power_percent_std_p50'][rec_idx])
        if enable_overclocking:
            oc_key = 'full_power_percent_oc_p50'
            if oc_key in results and results[oc_key] is not None and len(results[oc_key]) > rec_idx:
                h_oc = float(results[oc_key][rec_idx])

        util_cols = st.columns(2 if enable_overclocking else 1)
        with util_cols[0]:
            if cap_factor is not None and not np.isnan(cap_factor):
                st.metric("Utilization (Capacity Factor)", f"{cap_factor:.1f}%", help="Hashrate-weighted utilization across periods (P50).")
            else:
                st.metric("Utilization (Uptime)", f"{optimal['utilization']:.1f}%", help="Percent of periods with some power available.")
            if h_standard is not None:
                st.caption(f"≥ {installed_power_standard_kw/1000:.2f} MW for {h_standard:.1f}% of hours (simulated P50)")
        if enable_overclocking and len(util_cols) > 1 and h_oc is not None:
            with util_cols[1]:
                st.caption(f"≥ {installed_power_oc_kw/1000:.2f} MW for {h_oc:.1f}% of hours (simulated P50)")

        # Risk and performance metrics
        risk_col1, risk_col2, risk_col3 = st.columns(3)
        with risk_col1:
            st.metric("Risk Assessment", optimal['risk_assessment'], help="Qualitative assessment of investment risk.")
        with risk_col2:
            st.metric("Probability of Loss", f"{optimal['prob_loss']:.1%}", help="Chance of NPV being negative after the projection period.")
        with risk_col3:
            st.metric("Power Utilization", f"{optimal['utilization']:.1f}%", help="Percentage of time the fleet is actively mining (not idle due to zero power).")

        # --- Detailed Analysis Tabs ---
        st.subheader("📊 Detailed Analysis")
        tab1, tab2 = st.tabs(["Financial Projections (Optimal Fleet)", "NPV vs. Fleet Size"])

        with tab1:
            st.markdown(f"#### Projections for Recommended Fleet of {optimal['n_asics']} ASICs")
            
            # Find the index for the optimal fleet to get its median simulation data (with guard rails)
            fleet_sizes_res = results.get('fleet_sizes', []) or []
            details_list = results.get('median_simulation_details', []) or []

            # Try to find exact index; if missing, pick the closest available fleet size
            try:
                optimal_idx = fleet_sizes_res.index(optimal['n_asics'])
            except ValueError:
                if not fleet_sizes_res:
                    st.error("No fleet sizes present in simulation results.")
                    st.stop()
                # Fallback: choose the closest fleet size
                closest_idx = int(np.argmin([abs(s - optimal['n_asics']) for s in fleet_sizes_res]))
                optimal_idx = closest_idx

            # Bound check against details list length
            if optimal_idx >= len(details_list):
                st.warning("Result arrays are misaligned; using the closest available projection.")
                optimal_idx = max(0, len(details_list) - 1)

            median_details = details_list[optimal_idx]
            
            # Project economics for the optimal fleet
            projection_df, summary = project_mining_economics(
                median_details_data=median_details,
                n_asics=optimal['n_asics'],
                asic_price=asic_specs['unit_price'],
                annual_opex=annual_opex,
                projection_years=projection_years,
                discount_rate=discount_rate,
                additional_upfront_costs=additional_upfront_costs
            )
            
            # Display summary
            st.metric("Projected Total Profit", f"${summary['total_profit']:,.0f}")
            
            # Display formatted table
            st.dataframe(projection_df.style.format({
                'BTC Mined': '{:.4f}',
                'Revenue': '${:,.0f}',
                'Operating Costs': '${:,.0f}',
                'Net Income': '${:,.0f}',
                'Cumulative Cash Flow': '${:,.0f}',
                'Avg BTC Price': '${:,.0f}',
                'Avg Difficulty': '{:.2e}',
                'Energy Used (MWh)': '{:,.0f}',
                'Available Curtailed Energy (MWh)': '{:,.0f}',
                'Curtailed Energy Utilization (%)': '{:.1f}%'
            }))
            st.caption("Projections are based on the median simulation run for the optimal fleet size.")

            # Optional: Energy comparison chart per year (Used vs Available)
            if isinstance(median_details, dict) and 'energy_used_mwh' in median_details and 'available_energy_mwh' in median_details:
                # Summary metrics across the full projection horizon
                total_used_mwh = float(np.sum(median_details['energy_used_mwh']))
                total_available_mwh = float(np.sum(median_details['available_energy_mwh']))
                total_wasted_mwh = max(total_available_mwh - total_used_mwh, 0.0)
                util_pct_total = (total_used_mwh / total_available_mwh * 100.0) if total_available_mwh > 0 else 0.0
                # Simple aggregation check (tolerance for FP error)
                agg_ok = abs((total_used_mwh + total_wasted_mwh) - total_available_mwh) < 1e-6

                st.subheader("Energy Utilization Summary")
                sum_col1, sum_col2, sum_col3 = st.columns(3)
                with sum_col1:
                    st.metric("Total Energy Used (MWh)", f"{total_used_mwh:,.0f}")
                with sum_col2:
                    st.metric("Total Curtailed Energy (MWh)", f"{total_available_mwh:,.0f}")
                with sum_col3:
                    st.metric("% of Curtailed Energy Utilized", f"{util_pct_total:.1f}%")
                sum_col4, sum_col5 = st.columns(2)
                with sum_col4:
                    st.metric("Total Wasted Energy (MWh)", f"{total_wasted_mwh:,.0f}")
                with sum_col5:
                    st.metric("Aggregation Check", "PASS" if agg_ok else "FAIL")

                energy_years = list(range(1, projection_years + 1))
                energy_df = pd.DataFrame({
                    'Year': energy_years,
                    'Energy Used (MWh)': median_details['energy_used_mwh'],
                    'Available Curtailed Energy (MWh)': median_details['available_energy_mwh']
                })
                energy_df['Wasted Energy (MWh)'] = np.maximum(energy_df['Available Curtailed Energy (MWh)'] - energy_df['Energy Used (MWh)'], 0)
                energy_fig = go.Figure()
                energy_fig.add_trace(go.Bar(
                    x=energy_df['Year'], y=energy_df['Available Curtailed Energy (MWh)'],
                    name='Available Curtailed Energy (MWh)', marker_color='#9ecae1',
                    hovertemplate='<b>Year %{x}</b><br>Available: %{y:,.0f} MWh<extra></extra>'
                ))
                energy_fig.add_trace(go.Bar(
                    x=energy_df['Year'], y=energy_df['Energy Used (MWh)'],
                    name='Energy Used (MWh)', marker_color='#3182bd',
                    hovertemplate='<b>Year %{x}</b><br>Used: %{y:,.0f} MWh<extra></extra>'
                ))
                energy_fig.add_trace(go.Bar(
                    x=energy_df['Year'], y=energy_df['Wasted Energy (MWh)'],
                    name='Wasted Energy (MWh)', marker_color='#fdd0a2',
                    hovertemplate='<b>Year %{x}</b><br>Wasted: %{y:,.0f} MWh<extra></extra>'
                ))
                energy_fig.update_layout(
                    barmode='group',
                    xaxis_title='Year', yaxis_title='MWh', height=350,
                    yaxis=dict(tickformat=',.0f'),
                    legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1)
                )
                st.plotly_chart(energy_fig, use_container_width=True)
                st.info("Annual bars show Available, Used, and Wasted (Available−Used). Summary aggregates across years; Utilized % = Used / Available.")

                # Detailed per-year aggregation check
                with st.expander("Energy aggregation details (per year)", expanded=False):
                    per_year_df = energy_df.copy()
                    with np.errstate(divide='ignore', invalid='ignore'):
                        per_year_df['Utilization (%)'] = np.where(
                            per_year_df['Available Curtailed Energy (MWh)'] > 0,
                            per_year_df['Energy Used (MWh)'] / per_year_df['Available Curtailed Energy (MWh)'] * 100.0,
                            0.0
                        )
                    per_year_df['Check'] = np.isclose(
                        per_year_df['Energy Used (MWh)'] + per_year_df['Wasted Energy (MWh)'],
                        per_year_df['Available Curtailed Energy (MWh)'], atol=1e-6
                    )
                    st.dataframe(
                        per_year_df[['Year','Available Curtailed Energy (MWh)','Energy Used (MWh)','Wasted Energy (MWh)','Utilization (%)','Check']]
                        .style.format({
                            'Available Curtailed Energy (MWh)': '{:,.0f}',
                            'Energy Used (MWh)': '{:,.0f}',
                            'Wasted Energy (MWh)': '{:,.0f}',
                            'Utilization (%)': '{:.1f}%'
                        })
                    )

        with tab2:
            st.markdown("#### NPV Distribution Across Fleet Sizes")
            
            fig_npv = go.Figure()
            
            # P10 to P90 range
            fig_npv.add_trace(go.Scatter(
                x=results['fleet_sizes'],
                y=results['npv_p90'],
                fill=None,
                mode='lines',
                line_color='rgba(0,100,80,0.2)',
                name='P90 NPV'
            ))
            fig_npv.add_trace(go.Scatter(
                x=results['fleet_sizes'],
                y=results['npv_p10'],
                fill='tonexty', # Fill between P10 and P90
                mode='lines',
                line_color='rgba(0,100,80,0.2)',
                name='P10-P90 Range'
            ))
            
            # Expected NPV
            fig_npv.add_trace(go.Scatter(
                x=results['fleet_sizes'],
                y=results['npv_expected'],
                mode='lines+markers',
                name='Expected NPV',
                line=dict(color='rgb(0,100,80)', width=3)
            ))
            
            # Add vertical line for optimal fleet
            fig_npv.add_vline(
                x=optimal['n_asics'], 
                line_dash="dash", 
                line_color="red",
                annotation_text=f"Optimal: {optimal['n_asics']} ASICs"
            )
            
            fig_npv.update_layout(
                xaxis_title="Number of ASICs in Fleet",
                yaxis_title="Net Present Value ($)",
                height=450,
                hovermode='x unified'
            )
            st.plotly_chart(fig_npv, use_container_width=True)
            st.info("💡 This chart shows the range of likely NPV outcomes. The wider the shaded area, the higher the risk (uncertainty).")

    else:
        st.warning("Could not determine an optimal fleet size. This may be due to insufficient power or unfavorable economic conditions.")

# Footer
st.markdown("---")
st.caption("Bitcoin Mining Optimizer | Power-Constrained Operations with Dynamic Throttling")