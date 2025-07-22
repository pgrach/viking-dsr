# Hydro Bitcoin Mining Optimizer

A Streamlit web application for optimizing Bitcoin mining operations constrained by run-of-river hydroelectric power with dynamic ASIC throttling capabilities.

## Overview

This application helps determine the optimal number of Bitcoin mining ASICs to deploy given variable hydroelectric power availability. It uses Monte Carlo simulation to model mining economics under different market scenarios while accounting for power intermittency.

## Features

- **Power Profile Analysis**: Analyzes historical hydroelectric power data to understand availability patterns
- **Dynamic Throttling**: Models ASIC power adjustment (0-100%) based on available power
- **Monte Carlo Simulation**: Runs thousands of scenarios to estimate risk and returns
- **Fleet Optimization**: Tests different fleet sizes to find the optimal configuration
- **Economic Projections**: Projects revenue, costs, and profitability over multiple years
- **Risk Analysis**: Calculates VaR, probability of loss, and Sharpe ratios

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/hydro-bitcoin-miner.git
cd hydro-bitcoin-miner
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Ensure you have the required data files:
   - `hydro_flow.xlsx` - Historical hydroelectric power data
   - `btc_price.csv` - Historical Bitcoin prices
   - `btc_difficulty.csv` - Historical Bitcoin network difficulty

## Usage

1. Run the Streamlit app:
```bash
streamlit run app.py
```

2. Configure parameters in the sidebar:
   - ASIC specifications (model, hashrate, power, price)
   - Operating costs
   - Simulation parameters
   - Economic scenario

3. Click "Run Optimization" to start the analysis

4. Review results:
   - Optimal fleet size recommendation
   - NPV analysis by fleet size
   - Risk metrics and utilization analysis
   - Detailed financial projections

## File Structure

```
hydro-bitcoin-miner/
├── app.py                 # Streamlit web application
├── hydro_miner_model.py   # Core simulation and optimization logic
├── config.yaml           # Default configuration parameters
├── requirements.txt      # Python dependencies
├── hydro_flow.xlsx      # Hydroelectric power data (user provided)
├── btc_price.csv        # Bitcoin price data (user provided)
└── btc_difficulty.csv   # Bitcoin difficulty data (user provided)
```

## Configuration

Edit `config.yaml` to change default parameters:

- **ASIC specifications**: Model, hashrate, power consumption, price
- **Operating costs**: Annual operating expenses
- **Simulation parameters**: Number of simulations, projection years
- **Economic scenarios**: Difficulty growth and price change assumptions

## Model Assumptions

1. **Power Allocation**: When power is limited, all ASICs throttle equally
2. **No Backup Power**: Analysis assumes only hydroelectric power is available
3. **Constant Efficiency**: ASIC efficiency degrades linearly (5% per year)
4. **Transaction Fees**: Estimated at 5-10% additional revenue
5. **Pool Fees**: Typically 1-2% of mining revenue

## Optimization Approach

The model uses several techniques to find the optimal fleet size:

1. **Bootstrap Sampling**: Future power availability is simulated by sampling from historical data
2. **Monte Carlo Simulation**: Thousands of scenarios model different price and difficulty paths
3. **NPV Maximization**: Fleet size is optimized for maximum net present value
4. **Risk Adjustment**: Sharpe ratio and VaR metrics help balance risk and return

## Key Metrics

- **NPV (Net Present Value)**: Total value of the investment in today's dollars
- **IRR (Internal Rate of Return)**: Annualized return on investment
- **Payback Period**: Time to recover initial investment
- **Utilization Rate**: Percentage of time ASICs are operational
- **Capacity Factor**: Actual vs theoretical hashrate output

## Scenarios

Three pre-configured scenarios model different market conditions:

1. **Bear Market**: Low price growth, slower difficulty increase
2. **Base Case**: Moderate growth continuing current trends  
3. **Bull Market**: High price growth, rapid difficulty increase

## Tips for Use

1. Start with the base case scenario to establish a baseline
2. Test sensitivity by adjusting ASIC costs and operating expenses
3. Consider the P10-P90 NPV range, not just expected values
4. Pay attention to utilization rates - very large fleets may be underutilized
5. Factor in risk tolerance when choosing between NPV and risk-adjusted recommendations

## Future Enhancements

- Integration with real-time power and Bitcoin data APIs
- Support for multiple power sources (solar, grid backup)
- More sophisticated power allocation algorithms
- Machine learning for price/difficulty predictions
- Portfolio optimization across multiple sites

## License

MIT License - see LICENSE file for details

## Support

For questions or issues, please open a GitHub issue or contact support@example.com