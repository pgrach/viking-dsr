import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from scipy import stats
import warnings
from joblib import Parallel, delayed
warnings.filterwarnings('ignore')

# Import enhanced throttling for overclocking support
try:
    from enhanced_throttling import vectorized_enhanced_throttling
    ENHANCED_THROTTLING_AVAILABLE = True
except ImportError:
    ENHANCED_THROTTLING_AVAILABLE = False

def load_hydro_data(filepath):
    """Load and process hydroelectric power data from a half-hourly CSV file."""
    df = pd.read_csv(filepath)
    
    # The CSV might have 'Datetime' and 'Available Power (MW)' or the settlement format
    if 'Datetime' in df.columns:
        # Handle the format with 'Datetime' and 'Available Power (MW)'
        df = df.rename(columns={
            'Datetime': 'timestamp',
            'Available Power (MW)': 'available_power_mw'
        })
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df['available_power_kw'] = pd.to_numeric(df['available_power_mw'], errors='coerce').fillna(0) * 1000

    elif 'settlement_date' in df.columns:
        # Handle the format with 'settlement_date' and 'curtailed_mw'
        df['timestamp'] = pd.to_datetime(df['settlement_date']) + \
                          pd.to_timedelta((df['settlement_period'] - 1) * 30, unit='m')
        df['available_power_kw'] = pd.to_numeric(df['curtailed_mw'], errors='coerce').fillna(0) * 1000
    else:
        raise ValueError("Hydro data CSV is in an unknown format. Expected columns like 'Datetime' or 'settlement_date'.")

    df = df.dropna(subset=['timestamp'])
    df = df.set_index('timestamp')
    
    # Ensure a complete time series for the given period at 30-min frequency
    start_date = df.index.min()
    end_date = df.index.max()
    
    # Create a complete date range
    full_range = pd.date_range(start=start_date, end=end_date, freq='30min')
    
    # Reindex the dataframe to this full range, filling missing values with 0
    df = df.reindex(full_range, fill_value=0)
    
    # Reset index to have 'timestamp' as a column again
    df = df.reset_index().rename(columns={'index': 'timestamp'})
    
    # Keep only the necessary columns
    return df[['timestamp', 'available_power_kw']]

def load_btc_data(price_file, difficulty_file):
    """Load and process Bitcoin price and difficulty data."""
    # Load price data
    price_df = pd.read_csv(price_file)
    price_df['Date'] = pd.to_datetime(price_df['Date'])
    price_df = price_df.sort_values('Date')
    
    # Load difficulty data
    diff_df = pd.read_csv(difficulty_file)
    diff_df['Date'] = pd.to_datetime(diff_df['Date'])
    diff_df = diff_df.sort_values('Date')
    
    # Merge on date
    btc_df = pd.merge(price_df, diff_df, on='Date', how='inner')
    
    # Calculate mining economics
    btc_df['block_reward'] = get_block_reward(btc_df['Date'])
    btc_df['btc_per_th_day'] = (1e12 * 86400 * btc_df['block_reward']) / (btc_df['Difficulty'] * 2**32)
    btc_df['revenue_per_th_day'] = btc_df['btc_per_th_day'] * btc_df['Close']
    
    # Calculate growth rates
    btc_df['difficulty_growth_30d'] = btc_df['Difficulty'].pct_change(periods=30)
    btc_df['price_change_30d'] = btc_df['Close'].pct_change(periods=30)
    
    # Current values
    current = btc_df.iloc[-1]
    
    # Historical statistics
    last_year = btc_df[btc_df['Date'] > btc_df['Date'].max() - timedelta(days=365)].copy()
    
    # Calculate annualized growth rates
    if len(last_year) > 1:
        # Sort by date to ensure proper chronological order
        last_year = last_year.sort_values('Date')
        
        # Calculate annual price growth from oldest to newest
        price_start = last_year.iloc[0]['Close']  # Oldest price in last year
        price_end = last_year.iloc[-1]['Close']   # Newest price in last year
        
        # If we have enough data points (at least 350 days), calculate proper annual growth
        if len(last_year) >= 350:
            price_growth_annual = (price_end / price_start) - 1
        else:
            # If insufficient data, calculate from available data and annualize
            days_available = (last_year['Date'].max() - last_year['Date'].min()).days
            if days_available > 0:
                price_growth_raw = (price_end / price_start) - 1
                price_growth_annual = price_growth_raw * (365.0 / days_available)
            else:
                price_growth_annual = 0
        
        # For difficulty, calculate growth more robustly
        if len(last_year) >= 350:
            difficulty_start = last_year.iloc[0]['Difficulty']
            difficulty_end = last_year.iloc[-1]['Difficulty']
            difficulty_growth_annual = (difficulty_end / difficulty_start) - 1
        else:
            # Use average monthly changes if insufficient data
            valid_changes = last_year['difficulty_growth_30d'].dropna()
            if len(valid_changes) > 0:
                difficulty_growth_annual = valid_changes.mean() * 12
            else:
                difficulty_growth_annual = 0
        
        price_volatility_annual = last_year['Close'].pct_change().std() * np.sqrt(365)
    else:
        # Fallback if insufficient data
        price_growth_annual = 0
        difficulty_growth_annual = 0
        price_volatility_annual = 0
    
    return {
        'current_price': current['Close'],
        'current_difficulty': current['Difficulty'],
        'current_revenue_per_th': current['revenue_per_th_day'],
        'data_date': current['Date'],  # Add the actual date of the data
        'price_growth_annual': price_growth_annual,
        'difficulty_growth_annual': difficulty_growth_annual,
        'price_volatility_annual': price_volatility_annual,
        'historical_data': btc_df[['Date', 'Close', 'Difficulty', 'revenue_per_th_day']].rename(
            columns={'Date': 'date', 'Close': 'price', 'Difficulty': 'difficulty'})
    }

def get_block_reward(dates):
    """Get Bitcoin block reward for a given date or series of dates (vectorized)."""
    date_series = pd.to_datetime(dates)
    
    # Define halving dates and corresponding rewards
    halving_dates = [
        pd.to_datetime('2012-11-28'),
        pd.to_datetime('2016-07-09'),
        pd.to_datetime('2020-05-11'),
        pd.to_datetime('2024-04-20'),
        pd.to_datetime('2028-04-01')  # Approximate next halving
    ]
    rewards = [50, 25, 12.5, 6.25, 3.125, 1.5625]
    
    # Create conditions for np.select
    conditions = [
        date_series < halving_dates[0],
        (date_series >= halving_dates[0]) & (date_series < halving_dates[1]),
        (date_series >= halving_dates[1]) & (date_series < halving_dates[2]),
        (date_series >= halving_dates[2]) & (date_series < halving_dates[3]),
        (date_series >= halving_dates[3]) & (date_series < halving_dates[4])
    ]
    
    # Use np.select for efficient conditional logic
    return np.select(conditions, rewards[:-1], default=rewards[-1])

def analyze_power_profile(hydro_df):
    """Analyze hydroelectric power availability patterns from half-hourly data."""
    # Basic statistics
    power_data = hydro_df['available_power_kw'].dropna().values
    
    # Create a separate dataset for when the plant is operational
    operational_power_data = power_data[power_data > 0]
    
    # Calculate percentiles on the full dataset
    percentiles = {}
    for p in range(0, 101, 5):
        percentiles[p] = np.percentile(power_data, p)

    # Calculate percentiles on operational data only
    operational_percentiles = {}
    if operational_power_data.any():
        for p in range(0, 101, 5):
            operational_percentiles[p] = np.percentile(operational_power_data, p)
    else: # Handle case where there are no operational days
        for p in range(0, 101, 5):
            operational_percentiles[p] = 0

    # Create data for power duration curve (correcting the graph)
    duration_curve = {}
    for p in range(0, 101, 5):
        # For exceedance probability p, we need the (100-p)th percentile
        duration_curve[p] = np.percentile(power_data, 100 - p)
    
    # Calculate uptime based on periods where power is available
    uptime_percent = (power_data > 0).sum() / len(power_data) if len(power_data) > 0 else 0
    
    # Monthly averages
    hydro_df['month'] = hydro_df['timestamp'].dt.month
    monthly_avg_series = hydro_df.groupby('month')['available_power_kw'].mean()
    # Reindex to ensure we have all 12 months, filling missing ones with 0
    monthly_avg = monthly_avg_series.reindex(range(1, 13), fill_value=0).values.tolist()
    
    # Continuous run analysis (now in 30-min periods)
    runs = []
    current_run = 0
    for power in power_data:
        if power > 0:
            current_run += 1
        else:
            if current_run > 0:
                runs.append(current_run)
                current_run = 0
    if current_run > 0:
        runs.append(current_run)
    
    # Calculate statistics
    return {
        'avg_power_kw': np.mean(power_data),
        'min_power_kw': np.min(power_data),
        'max_power_kw': np.max(power_data),
        'p10_power_kw': percentiles[10],
        'p25_power_kw': percentiles[25],
        'p50_power_kw': percentiles[50],
        'p75_power_kw': percentiles[75],
        'p90_power_kw': percentiles[90],
        'op_avg_power_kw': np.mean(operational_power_data) if operational_power_data.any() else 0,
        'op_p10_power_kw': operational_percentiles[10],
        'op_p50_power_kw': operational_percentiles[50],
        'power_percentiles': duration_curve,
        'monthly_avg_power': monthly_avg,
        'uptime_percent': uptime_percent * 100,
        'zero_power_periods': (power_data == 0).sum(),
        'avg_run_periods': np.mean(runs) * 0.5 / 24 if runs else 0,  # Convert periods to days
        'median_run_periods': np.median(runs) * 0.5 / 24 if runs else 0, # Convert periods to days
        'power_distribution': power_data
    }

def simulate_power_availability(hydro_stats, n_periods, seed=None):
    """Bootstrap simulate future power availability from historical data for each period."""
    if seed is not None:
        np.random.seed(seed)
    
    # Use historical distribution of power available in each 30-min period
    power_dist = hydro_stats['power_distribution']
    
    # Simple bootstrap from the historical distribution
    simulated_power = np.random.choice(power_dist, size=n_periods, replace=True)
    
    return simulated_power

def calculate_mining_revenue(hashrate_th, power_available_kw, asic_power_kw, difficulty, btc_price, block_reward):
    """Calculate mining revenue with power constraints and throttling."""
    # Determine throttle level
    if power_available_kw == 0:
        return 0, 0, 0
    
    # Maximum ASICs that can run at full power
    max_asics_full = power_available_kw / asic_power_kw
    
    # If we have enough power, run at 100%
    if max_asics_full >= 1:
        effective_hashrate = hashrate_th
        power_used = asic_power_kw
    else:
        # Throttle to available power
        throttle_percent = max_asics_full
        effective_hashrate = hashrate_th * throttle_percent
        power_used = power_available_kw
    
    # Calculate BTC mined
    btc_per_day = (effective_hashrate * 1e12 * 86400 * block_reward) / (difficulty * 2**32)
    revenue = btc_per_day * btc_price
    return revenue, btc_per_day, power_used

def _precompute_simulation_parameters(projection_years, btc_data, scenario_params, sim_seed):
    """Pre-computes price, difficulty, and block rewards for each half-hour period of the simulation."""
    np.random.seed(sim_seed)
    
    n_periods = projection_years * 365 * 48
    n_months = projection_years * 12
    
    # Scenario parameters (monthly growth)
    diff_growth_monthly = (1 + scenario_params['difficulty_growth_annual'])**(1/12)
    price_trend_monthly = scenario_params['price_change_annual'] / 12
    price_volatility_annual = scenario_params.get('price_volatility_annual', btc_data['price_volatility_annual'])
    price_vol_monthly = price_volatility_annual / np.sqrt(12)

    # --- Generate monthly multipliers ---
    monthly_diff_multipliers = np.cumprod(np.insert(np.full(n_months, diff_growth_monthly), 0, 1))[:-1]
    monthly_shocks = np.random.normal(price_trend_monthly, price_vol_monthly, n_months)
    monthly_price_multipliers = np.cumprod(np.insert(1 + monthly_shocks, 0, 1))[:-1]

    # --- Create a timestamp for every half-hour period in the simulation ---
    base_date = pd.to_datetime(datetime.now())
    timestamps = base_date + pd.to_timedelta(np.arange(n_periods) * 30, unit='m')
    dates = pd.to_datetime(timestamps.date) # Get the date part for daily values

    # --- Map periods to months for accurate value projection ---
    month_indices = (timestamps.year - base_date.year) * 12 + (timestamps.month - base_date.month)
    month_indices = np.minimum(month_indices, n_months - 1)

    # --- Vectorized Difficulty and Price Calculation ---
    # These are now period-based, but the underlying growth is still monthly
    period_diff_multipliers = monthly_diff_multipliers[month_indices]
    period_difficulty = btc_data['current_difficulty'] * period_diff_multipliers

    period_price_multipliers = monthly_price_multipliers[month_indices]
    period_price = btc_data['current_price'] * period_price_multipliers
    np.maximum(period_price, 1000, out=period_price) # Floor price

    # --- Vectorized Block Reward Calculation (based on the date of each period) ---
    period_block_reward = get_block_reward(dates)
    
    return period_difficulty, period_price, period_block_reward

def _run_single_simulation(sim_seed, fleet_sizes_arr, asic_specs, scenario_params, btc_data, hydro_stats, projection_years, annual_opex, discount_rate, pool_fee, additional_upfront_costs):
    """
    Helper function to run a single vectorized simulation for all fleet sizes.
    This version is optimized for half-hourly periods and uses matrix operations.
    """
    # Extract ASIC parameters
    asic_hashrate = asic_specs['hash_rate_th']
    asic_power_kw = asic_specs['power_consumption_kw']
    asic_price = asic_specs['unit_price']
    n_periods = projection_years * 365 * 48

    # 1. Pre-computation of Time-Dependent Variables for each 30-min period
    period_difficulty, period_price, period_block_reward = _precompute_simulation_parameters(
        projection_years, btc_data, scenario_params, sim_seed
    )
    simulated_power = simulate_power_availability(hydro_stats, n_periods, seed=sim_seed)

    # 2. Full Vectorization Across Periods AND Fleet Sizes
    # Reshape period-based arrays for broadcasting against fleet arrays
    simulated_power_col = simulated_power[:, np.newaxis]
    period_difficulty_col = period_difficulty[:, np.newaxis]
    period_price_col = period_price[:, np.newaxis]
    period_block_reward_col = period_block_reward[:, np.newaxis]

    # Calculate fleet requirements (row vector)
    fleet_power_req_row = fleet_sizes_arr * asic_power_kw
    fleet_hashrate_row = fleet_sizes_arr * asic_hashrate

    # 3. Enhanced Power Management with Overclocking Support
    power_mask = simulated_power_col > 0
    
    if asic_specs.get('enable_overclocking', False) and ENHANCED_THROTTLING_AVAILABLE:
        # Use enhanced throttling that supports overclocking
        throttling_result = vectorized_enhanced_throttling(simulated_power_col, fleet_sizes_arr, asic_specs)
        effective_hashrate = throttling_result['effective_hashrate']
        fleet_power_avail = throttling_result['power_used']
    else:
        # Use standard throttling logic
        # Calculate available power for each fleet size (n_periods, n_fleets)
        fleet_power_avail = np.minimum(simulated_power_col, fleet_power_req_row)
        
        # Calculate throttle percentage for each fleet in each period
        throttle = np.divide(fleet_power_avail, fleet_power_req_row, 
                             out=np.zeros_like(fleet_power_avail), 
                             where=fleet_power_req_row > 0)

        # Calculate effective hashrate based on throttling
        effective_hashrate = fleet_hashrate_row * throttle
    
    # Calculate BTC mined in each 30-minute period for each fleet
    # Note: 86400 seconds/day is replaced with 1800 seconds/period
    btc_mined = (effective_hashrate * 1e12 * 1800 * period_block_reward_col) / (period_difficulty_col * 2**32)
    
    # Calculate revenue for each period
    period_revenue = (btc_mined * period_price_col) * (1 - pool_fee)
    # Ensure revenue is zero when power is off. Squeeze the mask to 1D for correct broadcasting.
    period_revenue[~power_mask.squeeze(), :] = 0

    # --- Energy accounting (kW to MWh over 30-min periods) ---
    # Fleet energy actually used each period for each fleet (periods x fleets)
    energy_used_mwh_period = (fleet_power_avail * 0.5) / 1000.0
    # Total curtailed energy available each period (periods x 1)
    available_energy_mwh_period = (simulated_power_col * 0.5) / 1000.0
    
    # Aggregate period revenues into annual cash flows
    annual_revenue = period_revenue.reshape(projection_years, 365 * 48, -1).sum(axis=1)
    
    # --- Additions for detailed projection ---
    annual_btc_mined = btc_mined.reshape(projection_years, 365 * 48, -1).sum(axis=1)
    annual_avg_price = period_price.reshape(projection_years, 365 * 48).mean(axis=1)
    annual_avg_difficulty = period_difficulty.reshape(projection_years, 365 * 48).mean(axis=1)
    annual_energy_used_mwh = energy_used_mwh_period.reshape(projection_years, 365 * 48, -1).sum(axis=1)
    annual_available_energy_mwh = available_energy_mwh_period.reshape(projection_years, 365 * 48, 1).sum(axis=1).squeeze()
    
    annual_details = {
        'revenue': annual_revenue,
        'btc_mined': annual_btc_mined,
        'avg_price': annual_avg_price,
        'avg_difficulty': annual_avg_difficulty,
        'energy_used_mwh': annual_energy_used_mwh,
        'available_energy_mwh': annual_available_energy_mwh,
    }
    
    # Calculate final cash flows including investment and opex
    initial_investment = -fleet_sizes_arr * asic_price - additional_upfront_costs
    cash_flows = np.zeros((projection_years + 1, len(fleet_sizes_arr)))
    cash_flows[0, :] = initial_investment
    cash_flows[1:, :] = annual_revenue - annual_opex

    # Calculate NPV for all fleet sizes
    discounts = np.array([(1 + discount_rate)**i for i in range(projection_years + 1)])
    npv = np.sum(cash_flows / discounts[:, np.newaxis], axis=0)

    # Calculate utilization and hashrate metrics
    periods_operational = (fleet_power_avail > 0).sum(axis=0)
    avg_utilization = (periods_operational / n_periods) * 100 if n_periods > 0 else 0
    total_effective_hashrate = effective_hashrate.sum(axis=0)

    # Calculate full power percentage for standard installed power
    full_power_intervals_std = (simulated_power_col >= fleet_power_req_row).sum(axis=0)
    total_intervals = simulated_power_col.shape[0]
    full_power_percent_std = (full_power_intervals_std / total_intervals) * 100

    # Calculate for OC installed power if applicable
    full_power_percent_oc = None
    if asic_specs.get('enable_overclocking', False) and 'power_consumption_kw_oc' in asic_specs:
        fleet_power_req_row_oc = fleet_sizes_arr * asic_specs['power_consumption_kw_oc']
        full_power_intervals_oc = (simulated_power_col >= fleet_power_req_row_oc).sum(axis=0)
        full_power_percent_oc = (full_power_intervals_oc / total_intervals) * 100

    return npv, cash_flows, avg_utilization, total_effective_hashrate, annual_details, full_power_percent_std, full_power_percent_oc

def calculate_irr(cash_flows):
    """Calculate Internal Rate of Return using numpy's IRR function"""
    try:
        # numpy.irr is deprecated, use numpy_financial.irr instead
        from numpy_financial import irr
        return irr(cash_flows) * 100  # Convert to percentage
    except:
        # Fallback to scipy optimization
        from scipy.optimize import brentq
        
        def npv_at_rate(rate, cash_flows):
            return sum(cf / (1 + rate)**i for i, cf in enumerate(cash_flows))
        
        try:
            # Find rate where NPV = 0
            irr_rate = brentq(lambda r: npv_at_rate(r, cash_flows), -0.99, 10.0)
            return irr_rate * 100
        except:
            return np.nan  # No valid IRR exists

def run_monte_carlo_simulation(hydro_stats, btc_data, asic_specs, annual_opex, 
                             n_simulations, fleet_sizes, scenario_params, projection_years, pool_fee, discount_rate, additional_upfront_costs):
    """Run Monte Carlo simulation for different fleet sizes."""
    import streamlit as st

    asic_hashrate = asic_specs['hash_rate_th']
    asic_power_kw = asic_specs['power_consumption_kw']

    if fleet_sizes is None or len(fleet_sizes) == 0:
        st.warning("No fleet sizes to simulate. Please provide a non-empty list of fleet sizes.")
        return {}
    fleet_sizes_arr = np.array(fleet_sizes)

    if n_simulations > 2000:
        st.warning(f"Reducing simulations from {n_simulations} to 500 for faster processing")
        n_simulations = 2000

    status_text = st.empty()
    status_text.text(f"Running {n_simulations} simulations across {len(fleet_sizes)} fleet sizes...")

    # Parallel execution
    results_list = Parallel(n_jobs=-1)(
        delayed(_run_single_simulation)(
            sim_seed=i,
            fleet_sizes_arr=fleet_sizes_arr,
            asic_specs=asic_specs,
            scenario_params=scenario_params,
            btc_data=btc_data,
            hydro_stats=hydro_stats,
            projection_years=projection_years,
            annual_opex=annual_opex,
            discount_rate=discount_rate,
            pool_fee=pool_fee,
            additional_upfront_costs=additional_upfront_costs
        ) for i in range(n_simulations)
    )

    status_text.text("Simulations complete. Aggregating results...")

    # Unpack results
    all_npvs = np.array([res[0] for res in results_list])
    all_cash_flows = np.array([res[1] for res in results_list])
    all_utilizations = np.array([res[2] for res in results_list])
    all_total_effective_hashrates = np.array([res[3] for res in results_list])
    all_annual_details = [res[4] for res in results_list]
    all_full_power_percents_std = np.array([res[5] for res in results_list])
    # OC may be None; build an array with NaNs where missing
    oc_list = [res[6] for res in results_list]
    if any(v is None for v in oc_list):
        # Create a NaN array with same shape as std and fill rows when available
        all_full_power_percents_oc = np.full_like(all_full_power_percents_std, np.nan)
        for i, v in enumerate(oc_list):
            if v is not None:
                all_full_power_percents_oc[i] = v
    else:
        all_full_power_percents_oc = np.array(oc_list)

    # --- New logic: Aggregate details across all simulations ---
    # Create arrays to hold all simulation details for easier aggregation
    # Shape: (n_simulations, n_years, n_fleets)
    all_revenues = np.array([d['revenue'] for d in all_annual_details])
    all_btc_mined = np.array([d['btc_mined'] for d in all_annual_details])
    all_energy_used_mwh = np.array([d['energy_used_mwh'] for d in all_annual_details])
    # Shape: (n_simulations, n_years)
    all_avg_prices = np.array([d['avg_price'] for d in all_annual_details])
    all_avg_difficulties = np.array([d['avg_difficulty'] for d in all_annual_details])
    all_available_energy_mwh = np.array([d['available_energy_mwh'] for d in all_annual_details])

    # Calculate median values across all simulations
    median_annual_revenue = np.percentile(all_revenues, 50, axis=0)
    median_annual_btc_mined = np.percentile(all_btc_mined, 50, axis=0)
    median_annual_energy_used_mwh = np.percentile(all_energy_used_mwh, 50, axis=0)
    median_avg_price = np.percentile(all_avg_prices, 50, axis=0)
    median_avg_difficulty = np.percentile(all_avg_difficulties, 50, axis=0)
    median_available_energy_mwh = np.percentile(all_available_energy_mwh, 50, axis=0)

    # Find cash flows from the median (P50) simulation for IRR/Payback
    median_npvs = np.percentile(all_npvs, 50, axis=0)
    median_simulation_cash_flows = []
    for i in range(len(fleet_sizes)):
        npvs_for_fleet = all_npvs[:, i]
        median_sim_idx = np.argmin(np.abs(npvs_for_fleet - median_npvs[i]))
        median_cash_flow = all_cash_flows[median_sim_idx, :, i]
        median_simulation_cash_flows.append(median_cash_flow.tolist())

    # --- Create the median simulation details for projection tables ---
    median_simulation_details = []
    for i in range(len(fleet_sizes)):
        details_for_fleet = {
            'revenue': median_annual_revenue[:, i],
            'btc_mined': median_annual_btc_mined[:, i],
            'avg_price': median_avg_price,
            'avg_difficulty': median_avg_difficulty,
            'energy_used_mwh': median_annual_energy_used_mwh[:, i],
            'available_energy_mwh': median_available_energy_mwh,
        }
        median_simulation_details.append(details_for_fleet)
    # --- End of new logic ---

    # Aggregate results
    # Capacity factor: average fraction of full hashrate delivered per period across the simulation
    # all_total_effective_hashrates is the sum over periods of effective hashrate (TH/s) for each simulation & fleet
    # Normalize by (max fleet hashrate TH/s) * (number of 30-min periods) to obtain a 0–100% percentage
    n_periods = projection_years * 365 * 48
    fleet_max_hashrate = fleet_sizes_arr * asic_hashrate  # TH/s per fleet at 100%
    denom = fleet_max_hashrate * n_periods  # broadcast over simulations
    capacity_factor = np.nanmean(
        np.divide(
            all_total_effective_hashrates,
            denom,
            out=np.zeros_like(all_total_effective_hashrates),
            where=denom > 0
        ),
        axis=0
    ) * 100
    capacity_factor = np.clip(capacity_factor, 0, 100)
    
    results = {
        'fleet_sizes': fleet_sizes,
        'npv_expected': np.mean(all_npvs, axis=0),
        'npv_p10': np.percentile(all_npvs, 10, axis=0),
        'npv_p50': median_npvs,
        'npv_p90': np.percentile(all_npvs, 90, axis=0),
        'prob_loss': np.mean(all_npvs < 0, axis=0),
        'var_95': -np.percentile(all_npvs, 5, axis=0),
        'sharpe_ratio': np.mean(all_npvs, axis=0) / np.std(all_npvs, axis=0),
        'avg_utilization': np.mean(all_utilizations, axis=0),
        'irr_median': [],
        'payback_months': [],
        'capacity_factor': capacity_factor,
    'median_simulation_cash_flows': median_simulation_cash_flows,
    'median_simulation_details': median_simulation_details,
    # Simulated future power: percentage of periods with available power >= installed power
    'full_power_percent_std_mean': np.nanmean(all_full_power_percents_std, axis=0),
    'full_power_percent_std_p50': np.nanpercentile(all_full_power_percents_std, 50, axis=0),
    'full_power_percent_oc_mean': np.nanmean(all_full_power_percents_oc, axis=0) if not np.isnan(all_full_power_percents_oc).all() else None,
    'full_power_percent_oc_p50': np.nanpercentile(all_full_power_percents_oc, 50, axis=0) if not np.isnan(all_full_power_percents_oc).all() else None,
    }

    # IRR and Payback from MEDIAN simulation
    for i, n_asics in enumerate(fleet_sizes):
        # Get the median simulation's cash flows
        median_sim_cash_flows = median_simulation_cash_flows[i]
        
        # Calculate IRR from this actual simulation
        irr_value = calculate_irr(median_sim_cash_flows)
        results['irr_median'].append(irr_value)
        
        # Payback (now also based on median simulation)
        cumulative_cfs = np.cumsum(median_sim_cash_flows)
        payback_year = np.where(cumulative_cfs > 0)[0]
        if payback_year.any():
            # Interpolate for more precise payback period
            first_positive_year = payback_year[0]
            if first_positive_year == 0:
                prev_cumulative_cf = 0
            else:
                prev_cumulative_cf = cumulative_cfs[first_positive_year - 1]
            year_cash_flow = median_sim_cash_flows[first_positive_year]
            
            months_into_year = (-prev_cumulative_cf / year_cash_flow) * 12 if year_cash_flow > 0 else 0
            total_months = ((first_positive_year - 1) * 12) + months_into_year
            results['payback_months'].append(total_months)
        else:
            results['payback_months'].append(np.nan) # Never paid back

    status_text.empty()
    return results

def calculate_payback_months(cash_flows):
    """Calculate payback period in months."""
    cumulative = 0
    for i, cf in enumerate(cash_flows):
        cumulative += cf
        if cumulative > 0:
            # Linear interpolation for month
            prev_cumulative = cumulative - cf
            months_in_year = 12 if i > 0 else 0
            month_fraction = (-prev_cumulative / cf) * 12 if cf != 0 else 0
            return (i - 1) * 12 + month_fraction
    return len(cash_flows) * 12

def calculate_actual_hashrate(cash_flows, n_asics, asic_hashrate, asic_price):
    """Estimate actual hashrate from cash flows (simplified)."""
    # This is a rough estimate - in practice would need more detailed tracking
    total_revenue = sum(cash_flows[1:])  # Exclude initial investment
    # Rough estimate assuming average conditions
    return total_revenue / (asic_price * n_asics) * asic_hashrate * 365

def calculate_optimal_fleet(results, hydro_stats, asic_specs):
    """Determine optimal fleet size based on simulation results, considering power constraints."""
    if not results or not results.get('fleet_sizes'):
        return {
            'n_asics': 0,
            'recommendation': "No simulation data available to determine optimal fleet.",
            'risk_assessment': "N/A"
        }

    # Calculate the practical limit of ASICs based on maximum installed power
    max_power_limit_asics = int(hydro_stats['max_power_kw'] / asic_specs['power_consumption_kw'])

    # Find fleet size with maximum expected NPV from the simulation
    max_npv_idx = np.argmax(results['npv_expected'])
    optimal_size_raw = results['fleet_sizes'][max_npv_idx]

    # The final recommended size is the smaller of the theoretical optimum and the practical power limit
    recommended_size = min(optimal_size_raw, max_power_limit_asics)
    
    # Find the index corresponding to the recommended size to pull other metrics
    if recommended_size in results['fleet_sizes']:
        recommended_idx = results['fleet_sizes'].index(recommended_size)
    else:
        # If the capped size is not a simulated step, find the closest one that is smaller
        feasible_sizes = [s for s in results['fleet_sizes'] if s <= recommended_size]
        if not feasible_sizes:
            return {
                'n_asics': 0,
                'recommendation': "No feasible fleet size found within power limits.",
                'risk_assessment': "Error"
            }
        recommended_size = max(feasible_sizes)
        recommended_idx = results['fleet_sizes'].index(recommended_size)

    # Determine recommendation text based on the new logic
    if recommended_size < optimal_size_raw:
        risk_assessment = "Power-Constrained Optimum"
        recommendation = (
            f"The recommended fleet of **{recommended_size} ASICs** is constrained by the maximum available power ({hydro_stats['max_power_kw']:.0f} kW). "
            f"While simulations show a theoretical peak NPV at {optimal_size_raw} ASICs, operating beyond the site's maximum power capacity is not possible."
        )
    else:
        if results['prob_loss'][recommended_idx] > 0.3:
            risk_assessment = "High Risk"
            recommendation = f"The optimal fleet of **{recommended_size} ASICs** carries a high risk with a >30% chance of loss. Consider a smaller, less risky fleet."
        elif results['prob_loss'][recommended_idx] > 0.1:
            risk_assessment = "Moderate Risk"
            recommendation = f"Proceed with the optimal fleet of **{recommended_size} ASICs**, but monitor market conditions closely due to moderate risk."
        else:
            risk_assessment = "Low Risk"
            recommendation = f"The optimal configuration of **{recommended_size} ASICs** offers the best risk-adjusted returns."
    
    return {
        'n_asics': recommended_size,
        'expected_npv': results['npv_expected'][recommended_idx],
        'npv_p10': results['npv_p10'][recommended_idx],
        'npv_p90': results['npv_p90'][recommended_idx],
        'irr_median': results['irr_median'][recommended_idx],
        'payback_months': results['payback_months'][recommended_idx],
        'prob_loss': results['prob_loss'][recommended_idx],
        'utilization': results['avg_utilization'][recommended_idx],
        'risk_assessment': risk_assessment,
        'recommendation': recommendation,
        'full_power_percent': 50,  # Placeholder
        'avg_throttle': 75,  # Placeholder
        'zero_production_days': 36  # Based on hydro stats
    }

def project_mining_economics(median_details_data, n_asics, asic_price, annual_opex, projection_years, discount_rate, additional_upfront_costs):
    """
    Generate a detailed projection table from a specific median simulation run.
    """
    if not median_details_data:
        return pd.DataFrame(), {}

    initial_investment = n_asics * asic_price + additional_upfront_costs
    
    # Create a DataFrame for years 1 to N
    df_years = pd.DataFrame({
        'Year': range(1, projection_years + 1),
        'BTC Mined': median_details_data['btc_mined'],
        'Revenue': median_details_data['revenue'],
        'Operating Costs': annual_opex,
        'Avg BTC Price': median_details_data['avg_price'],
        'Avg Difficulty': median_details_data['avg_difficulty'],
    })

    # Optional energy columns if provided
    if 'energy_used_mwh' in median_details_data:
        df_years['Energy Used (MWh)'] = median_details_data['energy_used_mwh']
    if 'available_energy_mwh' in median_details_data:
        df_years['Available Curtailed Energy (MWh)'] = median_details_data['available_energy_mwh']
    if 'Energy Used (MWh)' in df_years.columns and 'Available Curtailed Energy (MWh)' in df_years.columns:
        with np.errstate(divide='ignore', invalid='ignore'):
            util = np.divide(df_years['Energy Used (MWh)'], df_years['Available Curtailed Energy (MWh)']) * 100.0
            util = util.replace([np.inf, -np.inf], np.nan).fillna(0)
        df_years['Curtailed Energy Utilization (%)'] = util
    df_years['Net Income'] = df_years['Revenue'] - df_years['Operating Costs']
    
    # Create the initial investment row (Year 0)
    df_invest = pd.DataFrame({
        'Year': [0], 'BTC Mined': [np.nan], 'Revenue': [np.nan], 
        'Operating Costs': [np.nan], 'Net Income': [np.nan], 
        'Cumulative Cash Flow': [-initial_investment], 
        'Avg BTC Price': [np.nan], 'Avg Difficulty': [np.nan]
    })
    
    # Calculate cumulative cash flow for years 1 to N
    df_years['Cumulative Cash Flow'] = -initial_investment + df_years['Net Income'].cumsum()
    
    # Combine investment row with the rest of the data
    df = pd.concat([df_invest, df_years], ignore_index=True)
    
    # Calculate Annual Cash Flow for NPV verification
    df['Annual Cash Flow'] = df['Net Income']
    df.loc[0, 'Annual Cash Flow'] = -initial_investment
    df['Discounted Cash Flow'] = df['Annual Cash Flow'] / ((1 + discount_rate) ** df['Year'])
    npv = df['Discounted Cash Flow'].sum()

    # Create summary
    summary = {
        'total_btc_mined': df['BTC Mined'].sum(),
        'total_revenue': df['Revenue'].sum(),
        'total_profit': df['Cumulative Cash Flow'].iloc[-1],
        'npv': npv
    }

    # Select and reorder columns for display
    display_cols = [
        'Year', 'BTC Mined', 'Revenue', 'Operating Costs', 'Net Income', 
        'Cumulative Cash Flow', 'Avg BTC Price', 'Avg Difficulty'
    ]
    # Insert energy columns if present
    energy_cols = [col for col in [
        'Energy Used (MWh)', 'Available Curtailed Energy (MWh)', 'Curtailed Energy Utilization (%)'
    ] if col in df.columns]
    # Place energy columns after Operating Costs for readability
    if energy_cols:
        display_cols = ['Year', 'BTC Mined', 'Revenue', 'Operating Costs'] + energy_cols + [
            'Net Income', 'Cumulative Cash Flow', 'Avg BTC Price', 'Avg Difficulty'
        ]
    
    return df[display_cols], summary