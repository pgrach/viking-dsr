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
    Vectorized version of enhanced throttling for Monte Carlo simulation
    
    Args:
        simulated_power_col: Array of simulated power (n_days, 1)
        fleet_sizes_arr: Array of fleet sizes to test (1, n_fleets)
        asic_specs: ASIC specifications dictionary
        
    Returns:
        dict: Contains arrays for effective_hashrate, power_used, etc.
    """
    
    if not asic_specs.get('enable_overclocking', False):
        # Use existing standard logic
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
        # Enhanced overclocking logic
        base_power_kw = asic_specs['power_consumption_kw']
        base_hashrate = asic_specs['hash_rate_th']
        oc_power_kw = asic_specs['power_consumption_kw_oc']
        oc_hashrate = asic_specs['hash_rate_th_oc']
        
        # Calculate power requirements for each fleet size
        base_power_req_row = fleet_sizes_arr * base_power_kw
        oc_power_req_row = fleet_sizes_arr * oc_power_kw
        
        # Initialize output arrays
        n_days, n_fleets = simulated_power_col.shape[0], len(fleet_sizes_arr)
        effective_hashrate = np.zeros((n_days, n_fleets))
        power_used = np.zeros((n_days, n_fleets))
        
        for day in range(n_days):
            available_power = simulated_power_col[day, 0]
            
            for fleet_idx, n_asics in enumerate(fleet_sizes_arr):
                result = calculate_enhanced_throttling(available_power, n_asics, asic_specs)
                effective_hashrate[day, fleet_idx] = result['effective_hashrate']
                power_used[day, fleet_idx] = result['power_used']
        
        return {
            'effective_hashrate': effective_hashrate,
            'power_used': power_used,
            'throttle_factor': power_used / base_power_req_row
        }
