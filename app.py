# Streamlit App for Financial Simulation

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import random

# Set page configuration
st.set_page_config(
    page_title="Financial Simulation App",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Title and Introduction
st.title("Financial Simulation of Investment Strategies")
st.write("""
This app simulates investment strategies over randomly chosen time periods using historical data for QQQ and synthetic TQQQ. 
You can adjust the simulation parameters using the sidebar and visualize the results.
""")

# Sidebar Inputs
st.sidebar.header('Simulation Parameters')

num_simulations = st.sidebar.number_input(
    'Number of Simulations', min_value=10, max_value=500, value=100, step=10
)
period_length_years = st.sidebar.number_input(
    'Period Length (Years)', min_value=3, max_value=20, value=10, step=1
)
initial_investment = st.sidebar.number_input(
    'Initial Investment ($)', min_value=0, value=1000, step=100
)
monthly_investment = st.sidebar.number_input(
    'Monthly Investment ($)', min_value=0, value=100, step=10
)
qqq_threshold_percent = st.sidebar.slider(
    'QQQ Threshold (%)', min_value=0.0, max_value=10.0, value=3.3, step=0.1
)
tqqq_change_percent = st.sidebar.slider(
    'TQQQ Allocation Change (%)', min_value=0.0, max_value=20.0, value=5.0, step=0.1
)
expense_ratio_qqq_percent = st.sidebar.number_input(
    'QQQ Expense Ratio (%)', min_value=0.0, max_value=1.0, value=0.20, step=0.01
)
expense_ratio_tqqq_percent = st.sidebar.number_input(
    'TQQQ Expense Ratio (%)', min_value=0.0, max_value=2.0, value=0.95, step=0.01
)
capital_gains_tax_rate_percent = st.sidebar.slider(
    'Capital Gains Tax Rate (%)', min_value=0.0, max_value=50.0, value=15.0, step=0.1
)
rebalance_frequency = st.sidebar.selectbox(
    'Rebalance Frequency', options=['Monthly', 'Quarterly', 'Yearly']
)

# Convert percentages to decimals
qqq_threshold = qqq_threshold_percent / 100.0
tqqq_change = tqqq_change_percent / 100.0
expense_ratios = {
    'QQQ': expense_ratio_qqq_percent / 100.0,
    'TQQQ': expense_ratio_tqqq_percent / 100.0
}
capital_gains_tax_rate = capital_gains_tax_rate_percent / 100.0

# Function Definitions

@st.cache_data
def get_data(start, end):
    # Fetch QQQ data
    qqq = yf.download('QQQ', start=start, end=end)
    qqq = qqq['Adj Close']
    qqq.name = 'QQQ'  # Corrected line

    # Calculate daily returns for QQQ
    qqq_returns = qqq.pct_change().dropna()

    # Create synthetic TQQQ returns (approximate 3x leverage)
    tqqq_returns = qqq_returns * 3
    tqqq_returns[tqqq_returns > 1] = 1  # Limit returns to 100% per day
    tqqq_returns[tqqq_returns < -0.95] = -0.95  # Limit losses to 95% per day

    # Create synthetic TQQQ price series
    tqqq = (1 + tqqq_returns).cumprod() * 100  # Start from $100 for TQQQ
    tqqq.name = 'TQQQ'  # Corrected line

    # Combine QQQ and TQQQ data
    data = pd.concat([qqq, tqqq], axis=1).dropna()

    return data

def buy_and_hold(data, ticker, initial_investment, monthly_investment, expense_ratio, period_length_years):
    data = data.copy()
    data = data.reset_index()
    data['Date'] = pd.to_datetime(data['Date'])
    data['Time'] = (data['Date'] - data['Date'].iloc[0]).dt.days / 365.25  # Convert to years

    # Initialize portfolio
    portfolio_value = initial_investment
    total_investment = initial_investment
    units = initial_investment / data[ticker].iloc[0]

    portfolio_values = []
    for i, row in data.iterrows():
        # Monthly investment
        if i > 0 and row['Date'].day == 1:
            units += monthly_investment / row[ticker]
            total_investment += monthly_investment

        # Adjust for expense ratio (annual)
        daily_expense = expense_ratio / 252  # Approximate trading days per year
        units *= (1 - daily_expense)

        # Portfolio value
        portfolio_value = units * row[ticker]
        portfolio_values.append({
            'Date': row['Date'],
            'Time': row['Time'],
            'Portfolio Value': portfolio_value,
            'Total Investment': total_investment
        })

    return pd.DataFrame(portfolio_values)

def dynamic_allocation(data, initial_investment, monthly_investment, qqq_threshold, tqqq_change, expense_ratios, period_length_years, initial_allocation={'QQQ': 0.5, 'TQQQ': 0.5}):
    data = data.copy()
    data = data.reset_index()
    data['Date'] = pd.to_datetime(data['Date'])
    data['Time'] = (data['Date'] - data['Date'].iloc[0]).dt.days / 365.25  # Convert to years

    # Initialize portfolio
    portfolio_value = initial_investment
    total_investment = initial_investment
    allocation = {'QQQ': initial_allocation.get('QQQ', 0.5), 'TQQQ': initial_allocation.get('TQQQ', 0.5)}  # Make initial allocation configurable
    units = {
        'QQQ': (initial_investment * allocation['QQQ']) / data['QQQ'].iloc[0],
        'TQQQ': (initial_investment * allocation['TQQQ']) / data['TQQQ'].iloc[0]
    }

    portfolio_values = []
    for i, row in data.iterrows():
        # Daily returns
        qqq_return = row['QQQ'] / data['QQQ'].iloc[i - 1] - 1 if i > 0 else 0

        # Monthly investment
        if i > 0 and row['Date'].day == 1:
            # Allocate new investment based on current allocation
            units['QQQ'] += (monthly_investment * allocation['QQQ']) / row['QQQ']
            units['TQQQ'] += (monthly_investment * allocation['TQQQ']) / row['TQQQ']
            total_investment += monthly_investment

        # Adjust allocations based on QQQ daily return
        if abs(qqq_return) > qqq_threshold:
            change = tqqq_change if qqq_return > 0 else -tqqq_change
            allocation['TQQQ'] = min(max(allocation['TQQQ'] + change, 0), 1)
            allocation['QQQ'] = 1 - allocation['TQQQ']

            # Rebalance portfolio
            total_value = units['QQQ'] * row['QQQ'] + units['TQQQ'] * row['TQQQ']
            units['QQQ'] = (total_value * allocation['QQQ']) / row['QQQ']
            units['TQQQ'] = (total_value * allocation['TQQQ']) / row['TQQQ']

        # Adjust for expense ratios (annual)
        daily_expense_qqq = expense_ratios['QQQ'] / 252
        daily_expense_tqqq = expense_ratios['TQQQ'] / 252
        units['QQQ'] *= (1 - daily_expense_qqq)
        units['TQQQ'] *= (1 - daily_expense_tqqq)

        # Portfolio value
        portfolio_value = units['QQQ'] * row['QQQ'] + units['TQQQ'] * row['TQQQ']

        portfolio_values.append({
            'Date': row['Date'],
            'Time': row['Time'],
            'Portfolio Value': portfolio_value,
            'Total Investment': total_investment
        })

    return pd.DataFrame(portfolio_values)

def calculate_metrics(portfolio, period_length_years):
    portfolio = portfolio.copy()
    portfolio['Cumulative Return'] = portfolio['Portfolio Value'] / portfolio['Total Investment'] - 1

    # Final values
    final_value = portfolio['Portfolio Value'].iloc[-1]
    total_investment = portfolio['Total Investment'].iloc[-1]
    total_return = (final_value - total_investment) / total_investment

    # CAGR
    CAGR = (final_value / total_investment) ** (1 / period_length_years) - 1

    # Max Drawdown
    rolling_max = portfolio['Portfolio Value'].cummax()
    drawdowns = (portfolio['Portfolio Value'] - rolling_max) / rolling_max
    max_drawdown = drawdowns.min()

    # High Water Mark
    high_water_mark = rolling_max.max()

    # Volatility (Standard Deviation of Daily Returns)
    portfolio['Daily Return'] = portfolio['Portfolio Value'].pct_change()
    volatility = portfolio['Daily Return'].std() * np.sqrt(252)

    # Sharpe Ratio (Assuming Risk-Free Rate of 0%)
    sharpe_ratio = (portfolio['Daily Return'].mean() / portfolio['Daily Return'].std()) * np.sqrt(252)

    metrics = {
        'Final Portfolio Value': final_value,
        'Total Investment': total_investment,
        'Total Return (%)': total_return * 100,
        'CAGR (%)': CAGR * 100,
        'Max Drawdown (%)': max_drawdown * 100,
        'High Water Mark': high_water_mark,
        'Volatility (%)': volatility * 100,
        'Sharpe Ratio': sharpe_ratio
    }

    return metrics

def run_bootstrap_analysis(num_simulations, period_length_years, initial_investment, monthly_investment,
                           qqq_threshold, tqqq_change, expense_ratios, capital_gains_tax_rate, rebalance_frequency):
    # Fetch the full data range starting from when QQQ data is available
    data = get_data(start='1999-04-01', end='2023-01-01')
    data_dates = data.index

    # Determine the earliest and latest possible start dates
    total_days = int(period_length_years * 365.25)  # Account for leap years
    earliest_start = data_dates[0]
    latest_start = data_dates[-1] - pd.Timedelta(days=total_days)

    # Check if earliest_start is after latest_start
    if earliest_start > latest_start:
        st.write("Not enough data for the specified period length. Please adjust the period length or the date range.")
        return None, None

    # Store results
    results = []
    # Dictionaries to store time series data
    time_series_data = {
        'QQQ Buy & Hold': [],
        'TQQQ Buy & Hold': [],
        'Dynamic Allocation': []
    }

    simulations_completed = 0

    progress_bar = st.progress(0)

    while simulations_completed < num_simulations:
        # Randomly select a start date
        random_days = random.randint(0, (latest_start - earliest_start).days)
        random_start = earliest_start + pd.Timedelta(days=random_days)
        random_end = random_start + pd.Timedelta(days=total_days)

        # Ensure the end date is within the data range
        if random_end > data_dates[-1]:
            continue  # Skip this simulation if end date is beyond available data

        # Subset the data for the selected period
        period_data = data[(data.index >= random_start) & (data.index <= random_end)]

        # Skip if not enough data
        if len(period_data) < period_length_years * 252:  # Approximate trading days
            continue

        # Run strategies
        qqq_portfolio = buy_and_hold(
            period_data,
            'QQQ',
            initial_investment,
            monthly_investment,
            expense_ratio=expense_ratios['QQQ'],
            period_length_years=period_length_years
        )
        tqqq_portfolio = buy_and_hold(
            period_data,
            'TQQQ',
            initial_investment,
            monthly_investment,
            expense_ratio=expense_ratios['TQQQ'],
            period_length_years=period_length_years
        )
        dynamic_portfolio = dynamic_allocation(
            period_data,
            initial_investment=initial_investment,
            monthly_investment=monthly_investment,
            qqq_threshold=qqq_threshold,
            tqqq_change=tqqq_change,
            expense_ratios=expense_ratios,
            period_length_years=period_length_years
        )

        # Normalize time index
        max_time = period_length_years  # In years
        qqq_portfolio['Time'] = qqq_portfolio['Time'] / max_time * period_length_years
        tqqq_portfolio['Time'] = tqqq_portfolio['Time'] / max_time * period_length_years
        dynamic_portfolio['Time'] = dynamic_portfolio['Time'] / max_time * period_length_years

        # Store time series data
        time_series_data['QQQ Buy & Hold'].append({
            'Simulation': simulations_completed,
            'Portfolio': qqq_portfolio
        })
        time_series_data['TQQQ Buy & Hold'].append({
            'Simulation': simulations_completed,
            'Portfolio': tqqq_portfolio
        })
        time_series_data['Dynamic Allocation'].append({
            'Simulation': simulations_completed,
            'Portfolio': dynamic_portfolio
        })

        # Compute metrics
        for strategy_name, portfolio in [('QQQ Buy & Hold', qqq_portfolio),
                                         ('TQQQ Buy & Hold', tqqq_portfolio),
                                         ('Dynamic Allocation', dynamic_portfolio)]:

            metrics = calculate_metrics(portfolio, period_length_years)

            # Store results
            results.append({
                'Strategy': strategy_name,
                'Simulation': simulations_completed,
                'Start Date': random_start.date(),
                'End Date': random_end.date(),
                **metrics
            })

        simulations_completed += 1
        progress_bar.progress(simulations_completed / num_simulations)

    # Convert results to DataFrame
    results_df = pd.DataFrame(results)

    return results_df, time_series_data

def identify_highlights(results_df):
    highlights = {}

    for strategy in results_df['Strategy'].unique():
        strategy_df = results_df[results_df['Strategy'] == strategy]
        # Highest final portfolio value
        max_value_sim = strategy_df.loc[strategy_df['Final Portfolio Value'].idxmax()]['Simulation']
        # Lowest final portfolio value
        min_value_sim = strategy_df.loc[strategy_df['Final Portfolio Value'].idxmin()]['Simulation']
        # Median final portfolio value
        median_value = strategy_df['Final Portfolio Value'].median()
        median_value_sim = strategy_df.iloc[(strategy_df['Final Portfolio Value'] - median_value).abs().argsort()[:1]]['Simulation'].values[0]
        # Maximum drawdown
        max_drawdown_sim = strategy_df.loc[strategy_df['Max Drawdown (%)'].idxmin()]['Simulation']

        highlights[strategy] = {
            'Max Value': int(max_value_sim),
            'Min Value': int(min_value_sim),
            'Median Value': int(median_value_sim),
            'Max Drawdown': int(max_drawdown_sim)
        }

    return highlights

def find_global_y_range(time_series_data):
    all_values = []
    for strategy in time_series_data:
        for sim_data in time_series_data[strategy]:
            all_values.extend(sim_data['Portfolio']['Portfolio Value'].values)
    return min(all_values), max(all_values)

def plot_simulations_with_final_cagr(time_series_data, highlights, period_length_years, results_df):
    import numpy as np
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    # Create a subplot with 1 row and 3 columns
    fig = make_subplots(
        rows=1, cols=3, subplot_titles=("QQQ Buy & Hold", "TQQQ Buy & Hold", "Dynamic Allocation"),
        specs=[[{"secondary_y": False}, {"secondary_y": False}, {"secondary_y": False}]]
    )

    strategies = ['QQQ Buy & Hold', 'TQQQ Buy & Hold', 'Dynamic Allocation']

    # Find global min and max portfolio values across all simulations
    global_min_y, global_max_y = find_global_y_range(time_series_data)

    for idx, strategy in enumerate(strategies):
        for sim_data in time_series_data[strategy]:
            sim_num = sim_data['Simulation']
            portfolio = sim_data['Portfolio']
            time = portfolio['Time']  # Relative time (e.g., years)
            value = portfolio['Portfolio Value']

            # Determine the line color, width, and transparency
            if sim_num == highlights[strategy]['Max Value']:
                line = dict(color='green', width=2.5)
                name = 'Highest Final Value'
                line_opacity = 1.0
            elif sim_num == highlights[strategy]['Min Value']:
                line = dict(color='red', width=2.5)
                name = 'Lowest Final Value'
                line_opacity = 1.0
            elif sim_num == highlights[strategy]['Median Value']:
                line = dict(color='blue', width=2.5)
                name = 'Median Final Value'
                line_opacity = 1.0
            elif sim_num == highlights[strategy]['Max Drawdown']:
                line = dict(color='orange', width=2.5)
                name = 'Max Drawdown'
                line_opacity = 1.0
            else:
                line = dict(color='gray', width=0.5)
                name = None  # Don't add a name for the legend
                line_opacity = 0.2  # Make gray lines more transparent

            # Create hover text including start and end dates
            start_date = portfolio['Date'].iloc[0]
            end_date = portfolio['Date'].iloc[-1]
            hover_text = f"Start Date: {start_date}<br>End Date: {end_date}"

            # Plot portfolio value on the primary y-axis
            fig.add_trace(
                go.Scatter(
                    x=time,
                    y=value,
                    mode='lines',
                    line=line,
                    opacity=line_opacity,
                    showlegend=(name is not None and idx == 0),
                    name=name if name is not None and idx == 0 else None,
                    hoverinfo='x+y+text',
                    text=hover_text  # Include start and end dates in hover text
                ),
                row=1, col=idx + 1, secondary_y=False  # Primary y-axis for portfolio value
            )

    # Update layout
    fig.update_layout(
        height=500, width=1500, title_text="Portfolio Value Over Time for All Simulations",
        plot_bgcolor='white',  # Remove panel background color
        paper_bgcolor='white'  # Remove plot area background color
    )

    # Normalize the y-axis across panels
    for i in range(1, 4):
        # Update y-axes for portfolio value
        fig.update_yaxes(range=[global_min_y, global_max_y], title_text="Portfolio Value ($)", row=1, col=i, secondary_y=False)
        # Update x-axes to show relative time (years)
        fig.update_xaxes(title_text="Time (Years)", row=1, col=i)

    st.plotly_chart(fig, use_container_width=True)

# Run Simulations Button
if st.button('Run Simulations'):
    # Run the simulations
    with st.spinner('Running simulations...'):
        results_df, time_series_data = run_bootstrap_analysis(
            num_simulations=int(num_simulations),
            period_length_years=period_length_years,
            initial_investment=initial_investment,
            monthly_investment=monthly_investment,
            qqq_threshold=qqq_threshold,
            tqqq_change=tqqq_change,
            expense_ratios=expense_ratios,
            capital_gains_tax_rate=capital_gains_tax_rate,
            rebalance_frequency=rebalance_frequency
        )

    if results_df is not None:
        # Identify the simulations to highlight
        highlights = identify_highlights(results_df)

        # Summary statistics
        summary = results_df.groupby('Strategy').agg({
            'Total Return (%)': ['mean', 'std', 'min', 'max'],
            'CAGR (%)': ['mean', 'std', 'min', 'max'],
            'Max Drawdown (%)': ['mean', 'std', 'min', 'max'],
            'Volatility (%)': ['mean', 'std', 'min', 'max'],
            'Sharpe Ratio': ['mean', 'std', 'min', 'max']
        })

        # Display results
        st.header('Summary Statistics Across All Simulations')
        st.write(summary)

        # Plot the simulations
        plot_simulations_with_final_cagr(time_series_data, highlights, period_length_years, results_df)
    else:
        st.write("Simulation could not be completed. Please adjust the parameters.")
