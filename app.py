import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import yaml
from datetime import datetime, timedelta
from hydro_miner_model import (
    load_hydro_data, load_btc_data, analyze_power_profile,
    run_monte_carlo_simulation, calculate_optimal_fleet,
    project_mining_economics
)

# Page configuration
st.set_page_config(
    page_title="Hydro Bitcoin Mining Optimizer",
    page_icon="⚡",
    layout="wide"
)

# Load configuration
@st.cache_data
def load_config():
    with open('config.yaml', 'r') as file:
        return yaml.safe_load(file)

# Load data functions
@st.cache_data
def get_hydro_analysis():
    hydro_data = load_hydro_data('hydro_flow.csv')
    return analyze_power_profile(hydro_data)

@st.cache_data
def get_btc_data():
    return load_btc_data('btc_price.csv', 'btc_difficulty.csv')

# Clear cache if needed (uncomment the next line and reload if you see cache-related errors)
# get_btc_data.clear()

# Initialize session state
if 'simulation_results' not in st.session_state:
    st.session_state.simulation_results = None

# Header
st.title("⚡ Hydro Bitcoin Mining Optimizer")
st.markdown("*Optimize ASIC fleet size for run-of-river hydroelectric power*")

# Sidebar configuration
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # Load default config and data
    config = load_config()
    btc_data = get_btc_data()
    
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
        value=False,
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
            value=15.0,
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
        value=0, 
        min_value=0,
        help="Setup costs: installation, infrastructure, permits, etc."
    )
    
    st.subheader("🎲 Simulation Parameters")
    
    col_sim, col_step = st.columns(2)
    with col_sim:
        n_simulations = st.number_input(
            "Monte Carlo Runs", 
            value=min(config['simulation']['n_simulations'], 1000), 
            min_value=100, 
            max_value=2000,
            step=100,
            help="Number of simulation runs (more = accurate but slower)"
        )
    with col_step:
        fleet_step = st.number_input(
            "Fleet Size Step", 
            value=config['simulation']['fleet_step'], 
            min_value=1, 
            max_value=50,
            help="ASIC increment for testing different fleet sizes"
        )
    
    projection_years = st.number_input(
        "Projection Years", 
        value=config['simulation']['projection_years'], 
        min_value=1, 
        max_value=5,
        help="Investment time horizon for analysis"
    )
    
    st.subheader("📈 Economic Parameters")
    
    # Use only the custom scenario with historical defaults
    selected_scenario = config['scenarios']['custom'].copy()
    
    # Set defaults to historical values
    selected_scenario['difficulty_growth_annual'] = btc_data['difficulty_growth_annual']
    selected_scenario['price_change_annual'] = btc_data['price_growth_annual']
    selected_scenario['price_volatility_annual'] = btc_data['price_volatility_annual']
    
    st.markdown("**🎛️ Adjust Parameters (defaults based on historical 365-day data)**")
    
    col_diff, col_price = st.columns(2)
    with col_diff:
        custom_difficulty_growth = st.slider(
            "Difficulty Growth (%/year)",
            min_value=-50.0,
            max_value=150.0,
            value=selected_scenario['difficulty_growth_annual'] * 100,
            step=1.0,
            format="%.1f%%",
            help="Expected annual change in mining difficulty"
        ) / 100.0
    
    with col_price:
        custom_price_change = st.slider(
            "Price Change (%/year)",
            min_value=-50.0,
            max_value=150.0,
            value=selected_scenario['price_change_annual'] * 100,
            step=1.0,
            format="%.1f%%",
            help="Expected annual Bitcoin price growth"
        ) / 100.0
    
    custom_price_volatility = st.slider(
        "Price Volatility (%/year)",
        min_value=0.0,
        max_value=200.0,
        value=selected_scenario['price_volatility_annual'] * 100,
        step=5.0,
        format="%.1f%%",
        help="Expected annual price volatility (uncertainty)"
    ) / 100.0
    
    selected_scenario['difficulty_growth_annual'] = custom_difficulty_growth
    selected_scenario['price_change_annual'] = custom_price_change
    selected_scenario['price_volatility_annual'] = custom_price_volatility

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
    hydro_stats = get_hydro_analysis()
    
    # Add validation warning if power is very low
    if hydro_stats['max_power_kw'] < 50:
        st.warning("⚠️ **Low Power Alert**: Your hydro facility has very low power output. Consider reviewing the data or consulting with a power systems engineer.")
    
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
            help="Percentage of time the hydro plant generates any power"
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
        - **Note**: Plant uptime is {hydro_stats['uptime_percent']:.1f}%, so actual availability varies
        """)
    else:
        st.info(f"""
        💡 **Power Insights for {asic_model}**:
        - **Maximum capacity**: {max_asics} ASICs ({max_asics * asic_power_kw:.0f} kW)
        - **Reliable baseload**: {baseload_asics} ASICs can run 90% of the time when plant is ON (effective {baseload_asics * asic_power_kw / hydro_stats['max_power_kw'] * 100:.0f}% of max capacity)
        - **Power per ASIC**: {asic_power_kw:.1f} kW
        - **Note**: Plant uptime is {hydro_stats['uptime_percent']:.1f}%, so actual availability varies
        """)

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
    
    # Add ASIC power requirements with better styling
    colors = ['#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    asic_power_kw = asic_hashrate * asic_power / 1000
    max_possible_asics = int(hydro_stats['max_power_kw'] / asic_power_kw)
    
    # Show fleet lines up to max power capacity
    fleet_lines_to_show = list(range(fleet_step, max_possible_asics + 1, fleet_step))
    # Add the exact max if it's not already included
    if max_possible_asics not in fleet_lines_to_show and max_possible_asics > 0:
        fleet_lines_to_show.append(max_possible_asics)
    
    for i, n_asics in enumerate(fleet_lines_to_show):  # Show all fleet lines
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
    
    # Pre-simulation validation
    if enable_overclocking:
        # With overclocking, we test up to standard mode capacity but can run fewer in overclock mode
        max_possible_asics = int(hydro_stats['max_power_kw'] / (asic_hashrate * asic_power / 1000)) if asic_hashrate * asic_power > 0 else 0
        max_possible_asics_oc = int(hydro_stats['max_power_kw'] / (asic_hashrate_oc * asic_power_oc / 1000)) if asic_hashrate_oc * asic_power_oc > 0 else 0
        st.info(f"🔧 **Fleet Size Testing Range**: Will test 1 to {max_possible_asics} ASICs (max {max_possible_asics_oc} in pure overclock mode)")
    else:
        max_possible_asics = int(hydro_stats['max_power_kw'] / (asic_hashrate * asic_power / 1000)) if asic_hashrate * asic_power > 0 else 0
    
    # Show estimated runtime now that we have max_possible_asics
    estimated_time = (n_simulations * len(range(fleet_step, max_possible_asics + 1, fleet_step))) / 50000
    if estimated_time < 1:
        time_text = f"~{estimated_time*60:.0f} seconds"
    else:
        time_text = f"~{estimated_time:.1f} minutes"
    st.caption(f"⏱️ Estimated runtime: {time_text}")
    
    if max_possible_asics < fleet_step:
        st.error(f"❌ **Configuration Error**: Your fleet step ({fleet_step}) is larger than the maximum possible ASICs ({max_possible_asics}). Please reduce the fleet step in the sidebar.")
        st.stop()
    
    if max_possible_asics < 1:
        st.error("❌ **Power Insufficient**: Your hydro facility cannot power even a single ASIC. Consider using more efficient miners or verify your power data.")
        st.stop()
    
    # Enhanced progress display
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        with st.spinner("🔄 Initializing Monte Carlo simulation..."):
            status_text.text(f"🚀 Starting {n_simulations:,} simulations across {len(range(fleet_step, max_possible_asics + 1, fleet_step))} fleet sizes...")
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
            
            # Run simulation
            simulation_results = run_monte_carlo_simulation(
                hydro_stats=hydro_stats,
                btc_data=btc_data,
                asic_specs=asic_specs,
                annual_opex=annual_opex,
                n_simulations=n_simulations,
                fleet_step=fleet_step,
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
    
    # Create attractive metrics display
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric(
            "🎯 Recommended ASICs", 
            f"{optimal['n_asics']:,}",
            help="Optimal number of ASIC miners for maximum risk-adjusted returns"
        )
    with col2:
        npv_color = "normal" if optimal['expected_npv'] > 0 else "inverse"
        st.metric(
            "💰 Expected NPV", 
            f"${optimal['expected_npv']:,.0f}",
            delta=f"${optimal['expected_npv']:,.0f}" if optimal['expected_npv'] > 0 else None,
            delta_color=npv_color,
            help="Net Present Value - total profit in today's dollars"
        )
    with col3:
        irr_color = "normal" if optimal['irr_median'] > discount_rate * 100 else "inverse"
        st.metric(
            "📈 Median IRR", 
            f"{optimal['irr_median']:.1f}%",
            delta=f"vs {discount_rate*100:.1f}% target",
            delta_color=irr_color,
            help="Internal Rate of Return - annualized profit rate"
        )
    with col4:
        payback_color = "normal" if optimal['payback_months'] < 24 else "inverse"
        st.metric(
            "⏰ Payback Period", 
            f"{optimal['payback_months']:.1f} months",
            delta="Quick" if optimal['payback_months'] < 18 else "Slow" if optimal['payback_months'] > 36 else None,
            delta_color=payback_color,
            help="Time to recover initial investment"
        )
    
    # Add investment summary
    total_investment = optimal['n_asics'] * asic_specs['unit_price'] + additional_upfront_costs
    st.info(f"""
    💼 **Investment Summary**: 
    - **Total Investment**: ${total_investment:,.0f} ({optimal['n_asics']} × ${asic_specs['unit_price']:,.0f} + ${additional_upfront_costs:,.0f} setup)
    - **Risk Assessment**: {optimal['risk_assessment']}
    - **Power Usage**: {optimal['n_asics'] * asic_specs['power_consumption_kw']:.0f} kW ({optimal['n_asics'] * asic_specs['power_consumption_kw'] / hydro_stats['max_power_kw'] * 100:.1f}% of max capacity)
    """)
    
    # NPV by fleet size
    st.subheader("NPV Analysis by Fleet Size")
    
    fig_npv = go.Figure()
    
    # Add percentile bands
    fig_npv.add_trace(go.Scatter(
        x=results['fleet_sizes'],
        y=results['npv_p10'],
        fill=None,
        mode='lines',
        line_color='rgba(0,100,80,0)',
        showlegend=False
    ))
    
    fig_npv.add_trace(go.Scatter(
        x=results['fleet_sizes'],
        y=results['npv_p90'],
        fill='tonexty',
        mode='lines',
        line_color='rgba(0,100,80,0)',
        name='P10-P90 Range'
    ))
    
    fig_npv.add_trace(go.Scatter(
        x=results['fleet_sizes'],
        y=results['npv_expected'],
        mode='lines+markers',
        name='Expected NPV',
        line=dict(color='blue', width=3)
    ))
    
    # Mark optimal point
    fig_npv.add_trace(go.Scatter(
        x=[optimal['n_asics']],
        y=[optimal['expected_npv']],
        mode='markers',
        marker=dict(size=15, color='red', symbol='star'),
        name='Optimal'
    ))
    
    fig_npv.update_layout(
        xaxis_title="Number of ASICs",
        yaxis_title="Net Present Value ($)",
        height=400
    )
    st.plotly_chart(fig_npv, use_container_width=True)
    
    # Risk metrics
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Risk Metrics")

        risk_df = pd.DataFrame({
            'Fleet Size': results['fleet_sizes'],
            'Probability of Loss (%)': results['prob_loss'] * 100,
            'Value at Risk (95%)': results['var_95'],
            'Sharpe Ratio': results['sharpe_ratio']
        })

        # Format for display
        display_df = risk_df.copy()
        display_df['Probability of Loss (%)'] = display_df['Probability of Loss (%)'].map('{:.1f}%'.format)
        display_df['Value at Risk (95%)'] = display_df['Value at Risk (95%)'].map('${:,.0f}'.format)
        display_df['Sharpe Ratio'] = display_df['Sharpe Ratio'].map('{:.2f}'.format)

        st.dataframe(display_df.set_index('Fleet Size'))

        # Add VaR chart
        fig_var = px.line(risk_df, x='Fleet Size', y='Value at Risk (95%)',
                          title='Value at Risk (95%) by Fleet Size')
        fig_var.update_layout(height=250, yaxis_title="VaR ($)")
        st.plotly_chart(fig_var, use_container_width=True)

        # Add Sharpe Ratio chart
        fig_sharpe = px.line(risk_df, x='Fleet Size', y='Sharpe Ratio',
                             title='Sharpe Ratio by Fleet Size', color_discrete_sequence=['green'])
        fig_sharpe.update_layout(height=250, yaxis_title="Ratio")
        st.plotly_chart(fig_sharpe, use_container_width=True)
    
    with col2:
        st.subheader("Utilization Analysis")
        
        util_df = pd.DataFrame({
            'Fleet Size': results['fleet_sizes'],
            'Average Utilization (%)': results['avg_utilization'],
            'Capacity Factor (%)': results['capacity_factor']
        })
        
        fig_util = px.line(util_df, x='Fleet Size', y=['Average Utilization (%)', 'Capacity Factor (%)'],
                          title='Fleet Utilization Metrics', 
                          color_discrete_map={
                              'Average Utilization (%)': 'dodgerblue',
                              'Capacity Factor (%)': 'mediumorchid'
                          })
        fig_util.update_layout(height=300, yaxis_title="Percentage (%)")
        st.plotly_chart(fig_util, use_container_width=True)
    
    # Detailed projections based on the median simulation for the optimal fleet
    st.subheader(f"📈 Detailed Projections for {optimal['n_asics']} ASICs (Median Scenario)")

    optimal_n_asics = optimal['n_asics']
    if optimal_n_asics in results['fleet_sizes']:
        optimal_idx = results['fleet_sizes'].index(optimal_n_asics)
        median_details = results['median_simulation_details'][optimal_idx]
        
        proj_df, proj_summary = project_mining_economics(
            median_details_data=median_details,
            n_asics=optimal_n_asics,
            asic_price=asic_specs['unit_price'],
            annual_opex=annual_opex,
            projection_years=projection_years,
            discount_rate=discount_rate,
            additional_upfront_costs=additional_upfront_costs
        )

        # Format columns for display
        display_df = proj_df.copy()
        currency_cols = ['Revenue', 'Operating Costs', 'Net Income', 'Cumulative Cash Flow']
        for col in currency_cols:
            display_df[col] = display_df[col].map('${:,.0f}'.format)
        
        display_df['BTC Mined'] = display_df['BTC Mined'].map('{:.4f}'.format)
        display_df['Avg BTC Price'] = display_df['Avg BTC Price'].map('${:,.0f}'.format)
        display_df['Avg Difficulty'] = (display_df['Avg Difficulty'] / 1e12).map('{:.0f}T'.format)

        st.dataframe(display_df.set_index('Year'))

        # Summary report
        with st.expander("📋 Executive Summary", expanded=True):
            st.markdown(f"""
            ### Optimization Results for {selected_scenario['name']} Scenario
            
            **Recommended Configuration:**
            - Optimal Fleet Size: **{optimal['n_asics']} ASICs**
            - Total Investment: **${optimal['n_asics'] * asic_specs['unit_price'] + additional_upfront_costs:,.0f}**

            **Financial Metrics (Median Scenario):**
            - Median NPV (P50): **${results['npv_p50'][optimal_idx]:,.0f}**
            - Median IRR: **{optimal['irr_median']:.1f}%**
            - Payback Period: **{optimal['payback_months']:.1f} months**
            
            *These metrics are from the same representative simulation, ensuring complete consistency in the analysis.*

            **Key Insights:**
            - The recommended fleet size is based on maximizing the average NPV across all simulations.
            - The detailed projection below is from a single, representative simulation whose outcome was closest to the median (P50) NPV.
            - {optimal['risk_assessment']}
            
            **Power Utilization:**
            - Fleet will operate at 100% capacity {optimal['full_power_percent']:.1f}% of the time
            - Average throttling level: {optimal['avg_throttle']:.1f}%
            - Zero production expected {optimal['zero_production_days']:.1f}% of days
            
            **Financial Projections ({projection_years} years from Median Simulation):**
            - Total Revenue: ${proj_summary['total_revenue']:,.0f}
            - Net Profit: ${proj_summary['total_profit']:,.0f}
            - Verification NPV: **${proj_summary['npv']:,.0f}** (Matches the P50 NPV, confirming consistency)
            
            **Recommendation:** {optimal['recommendation']}

            ---
            *Note: This projection represents one specific simulation outcome (the median case). Actual results will vary based on Bitcoin price and mining difficulty changes.*
            """)
    else:
        st.warning("Could not generate detailed projection because the optimal fleet size was not in the simulated set.")

# Footer
st.markdown("---")
st.caption("Bitcoin Mining Optimizer | Hydroelectric Power Constraints with Dynamic Throttling")