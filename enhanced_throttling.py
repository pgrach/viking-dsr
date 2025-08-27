"""
Enhanced throttling functions that support overclocking capabilities
"""

import numpy as np

def calculate_enhanced_throttling(available_power_kw, n_asics, asic_specs):
    """
    Calculate optimal ASIC operation considering overclocking capabilities
    
    Args:
        available_power_kw: Available power in kW
        n_asics: Number of ASICs in fleet
        asic_specs: Dictionary containing ASIC specifications including overclocking
        
    Returns:
        dict: Contains effective_hashrate, power_used, operation_details
    """
    
    if not asic_specs.get('enable_overclocking', False):
        # Use standard throttling logic
        return calculate_standard_throttling(available_power_kw, n_asics, asic_specs)
    
    # Extract specifications
    base_hashrate = asic_specs['hash_rate_th']
    base_power_kw = asic_specs['power_consumption_kw']
    oc_hashrate = asic_specs['hash_rate_th_oc']
    oc_power_kw = asic_specs['power_consumption_kw_oc']
    
    # Calculate power requirements
    base_power_required = n_asics * base_power_kw
    oc_power_required = n_asics * oc_power_kw
    
    if available_power_kw <= 0:
        return {
            'effective_hashrate': 0,
            'power_used': 0,
            'throttle_factor': 0,
            'mode': 'offline',
            'asics_overclocked': 0,
            'asics_standard': 0,
            'asics_throttled': 0
        }
    
    elif available_power_kw >= oc_power_required:
        # Full overclocking possible
        return {
            'effective_hashrate': n_asics * oc_hashrate,
            'power_used': oc_power_required,
            'throttle_factor': 1.0,
            'mode': 'full_overclock',
            'asics_overclocked': n_asics,
            'asics_standard': 0,
            'asics_throttled': 0
        }
        
    elif available_power_kw >= base_power_required:
        # Mixed operation: some overclocked, some standard
        excess_power = available_power_kw - base_power_required
        additional_power_per_asic = oc_power_kw - base_power_kw
        
        # How many ASICs can be overclocked?
        asics_oc = min(n_asics, int(excess_power / additional_power_per_asic))
        asics_standard = n_asics - asics_oc
        
        effective_hashrate = (asics_oc * oc_hashrate + asics_standard * base_hashrate)
        power_used = (asics_oc * oc_power_kw + asics_standard * base_power_kw)
        
        return {
            'effective_hashrate': effective_hashrate,
            'power_used': power_used,
            'throttle_factor': available_power_kw / base_power_required,
            'mode': 'mixed_operation',
            'asics_overclocked': asics_oc,
            'asics_standard': asics_standard,
            'asics_throttled': 0
        }
        
    else:
        # Insufficient power for full standard operation - throttle
        throttle = available_power_kw / base_power_required
        
        return {
            'effective_hashrate': n_asics * base_hashrate * throttle,
            'power_used': available_power_kw,
            'throttle_factor': throttle,
            'mode': 'throttled_standard',
            'asics_overclocked': 0,
            'asics_standard': 0,
            'asics_throttled': n_asics
        }

def calculate_standard_throttling(available_power_kw, n_asics, asic_specs):
    """
    Standard throttling calculation (existing logic)
    """
    hashrate = asic_specs['hash_rate_th']
    power_kw = asic_specs['power_consumption_kw']
    required_power = n_asics * power_kw
    
    if available_power_kw <= 0:
        throttle = 0
    elif available_power_kw >= required_power:
        throttle = 1.0
    else:
        throttle = available_power_kw / required_power
    
    return {
        'effective_hashrate': n_asics * hashrate * throttle,
        'power_used': min(available_power_kw, required_power),
        'throttle_factor': throttle,
        'mode': 'standard' if throttle == 1.0 else 'throttled',
        'asics_overclocked': 0,
        'asics_standard': n_asics if throttle == 1.0 else 0,
        'asics_throttled': n_asics if throttle < 1.0 else 0
    }

def vectorized_enhanced_throttling(simulated_power_col, fleet_sizes_arr, asic_specs):
    """
    Vectorized version of enhanced throttling for Monte Carlo simulation.
    This version is fully vectorized using numpy to avoid slow Python loops.
    
    Args:
        simulated_power_col: Array of simulated power (n_periods, 1)
        fleet_sizes_arr: Array of fleet sizes to test (1, n_fleets)
        asic_specs: ASIC specifications dictionary
        
    Returns:
        dict: Contains arrays for effective_hashrate, power_used, etc.
    """
    
    if not asic_specs.get('enable_overclocking', False):
        # Use existing standard logic (this part is already vectorized)
        fleet_power_req_row = fleet_sizes_arr * asic_specs['power_consumption_kw']
        fleet_hashrate_row = fleet_sizes_arr * asic_specs['hash_rate_th']
        
        fleet_power_avail = np.minimum(simulated_power_col, fleet_power_req_row)
        throttle = np.divide(fleet_power_avail, fleet_power_req_row, 
                           out=np.zeros_like(fleet_power_avail), 
                           where=fleet_power_req_row > 0)
        
        effective_hashrate = fleet_hashrate_row * throttle
        
        return {
            'effective_hashrate': effective_hashrate,
            'power_used': fleet_power_avail,
            'throttle_factor': throttle
        }
    
    else:
        # Enhanced overclocking logic (fully vectorized)
        base_power_kw = asic_specs['power_consumption_kw']
        base_hashrate = asic_specs['hash_rate_th']
        oc_power_kw = asic_specs['power_consumption_kw_oc']
        oc_hashrate = asic_specs['hash_rate_th_oc']
        
        # Calculate power requirements for each fleet size (broadcastable)
        base_power_req_row = fleet_sizes_arr * base_power_kw
        oc_power_req_row = fleet_sizes_arr * oc_power_kw
        fleet_hashrate_row = fleet_sizes_arr * base_hashrate
        fleet_oc_hashrate_row = fleet_sizes_arr * oc_hashrate

        # --- Define conditions for np.select ---
        cond_offline = simulated_power_col <= 0
        cond_full_oc = simulated_power_col >= oc_power_req_row
        cond_mixed = (simulated_power_col >= base_power_req_row) & (simulated_power_col < oc_power_req_row)
        # The last condition (throttled_standard) will be the default
        
        conditions = [cond_offline, cond_full_oc, cond_mixed]

        # --- Define choices for effective_hashrate ---
        
        # Choice for mixed mode
        excess_power = simulated_power_col - base_power_req_row
        additional_power_per_asic = oc_power_kw - base_power_kw
        # Use np.floor and clip to ensure we don't get negative or too many OCs
        n_asics_oc = np.floor(excess_power / additional_power_per_asic)
        n_asics_oc = np.maximum(0, n_asics_oc)
        n_asics_oc = np.minimum(n_asics_oc, fleet_sizes_arr)
        n_asics_std = fleet_sizes_arr - n_asics_oc
        mixed_hashrate = n_asics_oc * oc_hashrate + n_asics_std * base_hashrate
        
        # Choice for throttled mode
        # The out array must match the broadcast shape of the operation
        broadcast_shape = (simulated_power_col.shape[0], fleet_sizes_arr.shape[0])
        out_array = np.zeros(broadcast_shape)
        throttle_factor = np.divide(simulated_power_col, base_power_req_row, out=out_array, where=base_power_req_row > 0)
        throttled_hashrate = fleet_hashrate_row * throttle_factor

        hashrate_choices = [
            0,                      # Offline
            fleet_oc_hashrate_row,  # Full OC
            mixed_hashrate          # Mixed
        ]
        effective_hashrate = np.select(conditions, hashrate_choices, default=throttled_hashrate)

        # --- Define choices for power_used ---
        
        # Power for mixed mode
        mixed_power = n_asics_oc * oc_power_kw + n_asics_std * base_power_kw
        
        # Power for throttled mode is all available power
        throttled_power = simulated_power_col

        power_choices = [
            0,                  # Offline
            oc_power_req_row,   # Full OC
            mixed_power         # Mixed
        ]
        power_used = np.select(conditions, power_choices, default=throttled_power)
        
        # Final check to ensure power used doesn't exceed available power
        power_used = np.minimum(power_used, simulated_power_col)
        
        return {
            'effective_hashrate': effective_hashrate,
            'power_used': power_used,
            'throttle_factor': np.divide(power_used, base_power_req_row, out=np.zeros_like(power_used), where=base_power_req_row > 0)
        }
