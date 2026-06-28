import streamlit as st
import pandas as pd
import numpy as np
import datetime
import plotly.graph_objects as go
import plotly.express as px
from scipy.stats import norm

# Import local modules
from utils import (
    download_stock_data,
    calculate_log_returns,
    perform_adf_test,
    calculate_rolling_volatility,
    get_descriptive_stats
)
from model import fit_garch, forecast_volatility

# Set Streamlit Page Configuration
st.set_page_config(
    page_title="Stock Price Volatility Forecasting",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Premium Design & Dark Aesthetics
st.markdown("""
<style>
    /* Google Font Import */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    /* Custom Styling for Metric Cards */
    div[data-testid="stMetric"] {
        background-color: #111520;
        border: 1px solid #1e293b;
        border-radius: 12px;
        padding: 18px 24px;
        box-shadow: 0 8px 16px rgba(0, 0, 0, 0.25);
        transition: transform 0.2s ease-in-out, border-color 0.2s ease-in-out;
    }
    div[data-testid="stMetric"]:hover {
        transform: translateY(-2px);
        border-color: #00f0ff;
    }
    div[data-testid="stMetric"] label {
        color: #94a3b8 !important;
        font-size: 0.85rem !important;
        font-weight: 600 !important;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    div[data-testid="stMetric"] div[data-testid="stMetricValue"] {
        color: #f8fafc !important;
        font-size: 1.8rem !important;
        font-weight: 700 !important;
        margin-top: 4px;
    }
    
    /* Styled container blocks */
    .dashboard-card {
        background-color: #0c101b;
        border: 1px solid #1e293b;
        border-radius: 16px;
        padding: 24px;
        margin-bottom: 24px;
        box-shadow: 0 10px 25px rgba(0,0,0,0.3);
    }
    
    /* Sidebar styling */
    section[data-testid="stSidebar"] {
        background-color: #070913;
        border-right: 1px solid #1e293b;
    }
    section[data-testid="stSidebar"] .stMarkdown h1 {
        font-size: 1.5rem;
        color: #00f0ff;
        font-weight: 700;
        margin-bottom: 20px;
    }
    
    /* Section headers */
    h1, h2, h3 {
        color: #f8fafc;
        font-weight: 700;
    }
    
    /* Accent text color */
    .text-accent {
        color: #00f0ff;
    }
    .text-success {
        color: #10b981;
    }
    .text-danger {
        color: #ef4444;
    }
    
    /* Highlight banners */
    .stat-badge {
        display: inline-block;
        padding: 4px 10px;
        border-radius: 6px;
        font-size: 0.75rem;
        font-weight: 700;
        text-transform: uppercase;
    }
    .stat-badge-stationary {
        background-color: rgba(16, 185, 129, 0.15);
        color: #10b981;
        border: 1px solid rgba(16, 185, 129, 0.3);
    }
    .stat-badge-nonstationary {
        background-color: rgba(239, 68, 68, 0.15);
        color: #ef4444;
        border: 1px solid rgba(239, 68, 68, 0.3);
    }
</style>
""", unsafe_allow_html=True)

# ----------------- SIDEBAR -----------------
st.sidebar.markdown("# 📈 VOLATILITY LAB")
st.sidebar.markdown("### Model Configuration")

# Stock Ticker Selection
ticker_option = st.sidebar.selectbox(
    "Select Stock Ticker",
    ["AAPL (Apple)", "TSLA (Tesla)", "MSFT (Microsoft)", "GOOGL (Alphabet)", "NVDA (NVIDIA)", "Custom Ticker..."]
)

if ticker_option == "Custom Ticker...":
    ticker = st.sidebar.text_input("Enter Ticker Symbol", value="SPY").strip().upper()
else:
    ticker = ticker_option.split(" ")[0]

# Date Range Input
five_years_ago = datetime.date.today() - datetime.timedelta(days=5*365)
start_date = st.sidebar.date_input("Start Date", five_years_ago)
end_date = st.sidebar.date_input("End Date", datetime.date.today())

# Forecast Horizon (trading days)
forecast_horizon = st.sidebar.slider(
    "Forecast Horizon (Trading Days)",
    min_value=5,
    max_value=90,
    value=30,
    step=5
)

# Rolling Volatility Window
rolling_window = st.sidebar.slider(
    "Rolling Volatility Window (Days)",
    min_value=5,
    max_value=126,
    value=21,
    step=1
)

# GARCH Distribution Selection
garch_dist = st.sidebar.selectbox(
    "GARCH Innovation Distribution",
    options=["normal", "t", "skewt"],
    format_func=lambda x: {
        "normal": "Normal (Gaussian)",
        "t": "Student's t (Fat Tails)",
        "skewt": "Skewed Student's t (Asymmetric Fat Tails)"
    }[x]
)

# Input Validation
if start_date >= end_date:
    st.error("Error: Start Date must be prior to End Date.")
    st.stop()

# ----------------- MAIN CONTENT -----------------
st.markdown(f"# 📈 Stock Price Volatility Forecasting: <span class='text-accent'>{ticker}</span>", unsafe_allow_html=True)
st.markdown("Analyze stock return characteristics, perform stationarity diagnostics, and fit/forecast conditional volatility using the **GARCH(1,1)** model.")

# Load Data
with st.spinner("Fetching stock historical data..."):
    try:
        df_stock = download_stock_data(ticker, start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d"))
    except Exception as e:
        st.error(f"Failed to fetch data for ticker **{ticker}**: {str(e)}")
        st.info("Tip: Double check the symbol on Yahoo Finance (e.g., ^GSPC for S&P 500, BTC-USD for Bitcoin).")
        st.stop()

# Calculate returns
close_col = "Adj Close" if "Adj Close" in df_stock.columns else "Close"
stock_prices = df_stock[close_col]
log_returns = calculate_log_returns(stock_prices)

# Perform Calculations
stats = get_descriptive_stats(log_returns)
adf_results = perform_adf_test(log_returns)
rolling_vol = calculate_rolling_volatility(log_returns, window=rolling_window)

# Fit GARCH model
with st.spinner("Training GARCH(1,1) model..."):
    try:
        garch_fit = fit_garch(log_returns, dist=garch_dist)
        forecast_df = forecast_volatility(garch_fit, horizon=forecast_horizon)
    except Exception as e:
        st.error(f"Failed to fit GARCH(1,1) model: {str(e)}")
        st.stop()

# ----------------- KEY METRICS ROW -----------------
col1, col2, col3, col4 = st.columns(4)

with col1:
    ann_ret_pct = stats["annualized_mean"] * 100
    st.metric(
        label="Annualized Mean Return",
        value=f"{ann_ret_pct:.2f}%",
        delta=f"Daily: {stats['daily_mean']*100:.3f}%"
    )

with col2:
    ann_vol_pct = stats["annualized_std"] * 100
    st.metric(
        label="Annualized Volatility (Hist)",
        value=f"{ann_vol_pct:.2f}%",
        delta=f"Daily Std: {stats['daily_std']*100:.3f}%"
    )

with col3:
    # Stationarity indicator
    status_label = "Stationary (Pass)" if adf_results["is_stationary"] else "Non-Stationary (Fail)"
    status_badge = f"<span class='stat-badge stat-badge-stationary'>{status_label}</span>" if adf_results["is_stationary"] else f"<span class='stat-badge stat-badge-nonstationary'>{status_label}</span>"
    
    st.metric(
        label="ADF Test p-value",
        value=f"{adf_results['p_value']:.4e}",
        delta=None
    )
    st.markdown(f"**Stationarity (ADF):** {status_badge}", unsafe_allow_html=True)

with col4:
    persistence_pct = garch_fit["persistence"] * 100
    st.metric(
        label="GARCH Volatility Persistence",
        value=f"{persistence_pct:.1f}%",
        delta="Unstable (>=100%)" if garch_fit["persistence"] >= 1.0 else "Stable (<100%)",
        delta_color="normal" if garch_fit["persistence"] < 1.0 else "inverse"
    )

st.markdown("---")

# ----------------- TABS CREATION -----------------
tab_eda, tab_rolling, tab_model = st.tabs([
    "📊 Exploratory Data Analysis",
    "🔄 Rolling Volatility & Clustering",
    "🧬 GARCH(1,1) Fit & Forecast"
])

# ----------------- TAB 1: EDA -----------------
with tab_eda:
    st.markdown("### Stock Price & Log Returns Distribution")
    
    eda_col1, eda_col2 = st.columns(2)
    
    with eda_col1:
        # Closing Price Chart
        fig_price = go.Figure()
        fig_price.add_trace(go.Scatter(
            x=stock_prices.index,
            y=stock_prices.values,
            mode='lines',
            name=ticker,
            line=dict(color='#00f0ff', width=2),
            hovertemplate="<b>Date:</b> %{x}<br><b>Price:</b> $%{y:,.2f}<extra></extra>"
        ))
        fig_price.update_layout(
            title=f"{ticker} Historical Close Price",
            xaxis_title="Date",
            yaxis_title="Price ($)",
            template="plotly_dark",
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            hovermode="x unified",
            xaxis=dict(showgrid=True, gridcolor="#1e293b"),
            yaxis=dict(showgrid=True, gridcolor="#1e293b")
        )
        st.plotly_chart(fig_price, use_container_width=True)
        
    with eda_col2:
        # Histogram with fitted normal curve
        fig_hist = go.Figure()
        
        # Empirical log returns in %
        ret_pct = log_returns * 100
        
        fig_hist.add_trace(go.Histogram(
            x=ret_pct,
            histnorm='probability density',
            name='Empirical returns',
            marker=dict(color='rgba(108, 92, 231, 0.6)', line=dict(color='rgba(108, 92, 231, 1)', width=1)),
            opacity=0.75,
            nbinsx=100
        ))
        
        # Compute Normal distribution fit
        x_range = np.linspace(ret_pct.min(), ret_pct.max(), 500)
        norm_pdf = norm.pdf(x_range, ret_pct.mean(), ret_pct.std())
        
        fig_hist.add_trace(go.Scatter(
            x=x_range,
            y=norm_pdf,
            mode='lines',
            name='Fitted Normal',
            line=dict(color='#ff007f', width=2, dash='dash')
        ))
        
        fig_hist.update_layout(
            title=f"Distribution of Daily Log Returns vs. Normal Fit",
            xaxis_title="Daily Log Return (%)",
            yaxis_title="Density",
            template="plotly_dark",
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            legend=dict(x=0.02, y=0.98),
            xaxis=dict(showgrid=True, gridcolor="#1e293b"),
            yaxis=dict(showgrid=True, gridcolor="#1e293b")
        )
        st.plotly_chart(fig_hist, use_container_width=True)
        
    # Statistical Insights Table
    st.markdown("#### Return Distribution Statistics")
    
    desc_df = pd.DataFrame({
        "Statistic": [
            "Sample Size (Trading Days)",
            "Daily Mean Return (%)",
            "Annualized Mean Return (%)",
            "Daily Volatility / Std Dev (%)",
            "Annualized Volatility (%)",
            "Skewness (Asymmetry)",
            "Excess Kurtosis (Fat Tails)",
            "Minimum Return (%)",
            "Maximum Return (%)"
        ],
        "Value": [
            f"{len(log_returns):,}",
            f"{stats['daily_mean']*100:.4f}%",
            f"{stats['annualized_mean']*100:.2f}%",
            f"{stats['daily_std']*100:.4f}%",
            f"{stats['annualized_std']*100:.2f}%",
            f"{stats['skewness']:.4f}",
            f"{stats['kurtosis']:.4f}", # pandas kurtosis returns excess kurtosis
            f"{stats['min']*100:.2f}%",
            f"{stats['max']*100:.2f}%"
        ]
    })
    
    st.dataframe(desc_df, use_container_width=True, hide_index=True)
    
    # Interpretation box
    st.markdown("""
    > **💡 Distribution Analysis:**  
    > Financial log returns typically exhibit **fat-tails (leptokurtosis)**: excess kurtosis is positive, and return distributions have more values clustered at the extreme ends compared to a normal distribution. This results in the red dashed normal curve under-representing the extreme shocks, highlighting the need for volatility models like GARCH that account for changing variance over time.
    """)

# ----------------- TAB 2: ROLLING VOL & CLUSTERING -----------------
with tab_rolling:
    st.markdown("### Volatility Clustering Analysis")
    
    # Returns Timeline Chart
    fig_ret = go.Figure()
    fig_ret.add_trace(go.Scatter(
        x=log_returns.index,
        y=log_returns.values * 100,
        mode='lines',
        name='Daily Return',
        line=dict(color='rgba(255, 255, 255, 0.45)', width=1)
    ))
    
    fig_ret.update_layout(
        title="Daily Log Returns (%) - Showing Volatility Clustering",
        xaxis_title="Date",
        yaxis_title="Return (%)",
        template="plotly_dark",
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(showgrid=True, gridcolor="#1e293b"),
        yaxis=dict(showgrid=True, gridcolor="#1e293b")
    )
    st.plotly_chart(fig_ret, use_container_width=True)
    
    # Rolling Volatility Chart
    fig_roll = go.Figure()
    fig_roll.add_trace(go.Scatter(
        x=rolling_vol.index,
        y=rolling_vol["Annualized"] * 100,
        mode='lines',
        name=f"Annualized Rolling Vol ({rolling_window}d)",
        line=dict(color='#ff7f00', width=2)
    ))
    
    fig_roll.update_layout(
        title=f"Annualized Rolling Volatility ({rolling_window}-Day Window)",
        xaxis_title="Date",
        yaxis_title="Annual Volatility (%)",
        template="plotly_dark",
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(showgrid=True, gridcolor="#1e293b"),
        yaxis=dict(showgrid=True, gridcolor="#1e293b")
    )
    st.plotly_chart(fig_roll, use_container_width=True)
    
    st.markdown(f"""
    > **💡 Volatility Clustering Explanation:**  
    > Notice how periods of large returns (either positive or negative) cluster together, and periods of small returns cluster together. This is **volatility clustering** (Mandelbrot, 1963). The rolling standard deviation in orange confirms that volatility rises sharply in response to market shocks and decays slowly over time.
    """)

# ----------------- TAB 3: GARCH FIT & FORECAST -----------------
with tab_model:
    st.markdown("### GARCH(1,1) Volatility Modeling")
    
    m_col1, m_col2 = st.columns([2, 1])
    
    with m_col1:
        # Comparison of Returns and Conditional Volatility
        fig_cond = go.Figure()
        
        # Add Returns (faded)
        fig_cond.add_trace(go.Scatter(
            x=log_returns.index,
            y=log_returns.values * 100,
            mode='lines',
            name='Daily Log Returns (%)',
            line=dict(color='rgba(148, 163, 184, 0.3)', width=1)
        ))
        
        # Add GARCH Conditional Volatility (Annualized)
        fig_cond.add_trace(go.Scatter(
            x=garch_fit["fitted_volatility_annualized"].index,
            y=garch_fit["fitted_volatility_annualized"].values * 100,
            mode='lines',
            name='GARCH Conditional Volatility (Annualized %)',
            line=dict(color='#00ffd0', width=2)
        ))
        
        fig_cond.update_layout(
            title="GARCH(1,1) Conditional Volatility (Annualized) vs. Daily Returns",
            xaxis_title="Date",
            yaxis_title="Volatility / Return (%)",
            template="plotly_dark",
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            legend=dict(x=0.02, y=0.98),
            xaxis=dict(showgrid=True, gridcolor="#1e293b"),
            yaxis=dict(showgrid=True, gridcolor="#1e293b")
        )
        st.plotly_chart(fig_cond, use_container_width=True)
        
    with m_col2:
        # Model parameter summary table
        st.markdown("#### Model Parameters")
        params_df = garch_fit["parameters"].copy()
        
        # Format columns for display
        params_df["Coefficient"] = params_df["Coefficient"].map('{:,.4f}'.format)
        params_df["Std Error"] = params_df["Std Error"].map('{:,.4f}'.format)
        params_df["t-Statistic"] = params_df["t-Statistic"].map('{:,.2f}'.format)
        params_df["p-Value"] = params_df["p-Value"].map('{:.4e}'.format)
        
        st.dataframe(params_df, use_container_width=True)
        
        # Model stats
        st.markdown("#### Model Selection Criteria")
        sel_df = pd.DataFrame({
            "Metric": ["Log-Likelihood", "AIC (Akaike Info Criterion)", "BIC (Bayesian Info Criterion)"],
            "Value": [
                f"{garch_fit['log_likelihood']:,.2f}",
                f"{garch_fit['aic']:,.2f}",
                f"{garch_fit['bic']:,.2f}"
            ]
        })
        st.dataframe(sel_df, use_container_width=True, hide_index=True)

    st.markdown("---")
    st.markdown(f"### Volatility Forecast ({forecast_horizon} Trading Days Ahead)")
    
    fore_col1, fore_col2 = st.columns([2, 1])
    
    with fore_col1:
        # Plot Forecasted Volatility
        fig_fore = go.Figure()
        
        # Forecast path
        fig_fore.add_trace(go.Scatter(
            x=forecast_df.index,
            y=forecast_df["Annualized Volatility (%)"],
            mode='lines+markers',
            name='Forecasted Volatility',
            line=dict(color='#ff007f', width=3),
            marker=dict(size=6, symbol='circle')
        ))
        
        # Add Horizontal Line for Unconditional Volatility
        uncond_vol_annual = garch_fit["unconditional_volatility_annualized"]
        if not np.isnan(uncond_vol_annual):
            fig_fore.add_trace(go.Scatter(
                x=[forecast_df.index[0], forecast_df.index[-1]],
                y=[uncond_vol_annual * 100, uncond_vol_annual * 100],
                mode='lines',
                name='Long-Run Unconditional Volatility',
                line=dict(color='#10b981', width=2, dash='dash')
            ))
            
        fig_fore.update_layout(
            title=f"GARCH(1,1) Volatility Forecast (Annualized %)",
            xaxis_title="Forecast Date",
            yaxis_title="Annualized Volatility (%)",
            template="plotly_dark",
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            legend=dict(x=0.02, y=0.98),
            xaxis=dict(showgrid=True, gridcolor="#1e293b"),
            yaxis=dict(showgrid=True, gridcolor="#1e293b")
        )
        st.plotly_chart(fig_fore, use_container_width=True)
        
    with fore_col2:
        st.markdown("#### Forecast Output Data")
        
        # Display forecast dataframe
        display_fore = forecast_df[["Day", "Daily Volatility (%)", "Annualized Volatility (%)"]].copy()
        display_fore["Daily Volatility (%)"] = display_fore["Daily Volatility (%)"].map('{:,.4f}%'.format)
        display_fore["Annualized Volatility (%)"] = display_fore["Annualized Volatility (%)"].map('{:,.2f}%'.format)
        
        st.dataframe(display_fore, height=300, use_container_width=True)
        
        # Download forecast CSV option
        csv_data = forecast_df.to_csv()
        st.download_button(
            label="📥 Download Forecast Results (CSV)",
            data=csv_data,
            file_name=f"{ticker}_garch_volatility_forecast_{forecast_horizon}d.csv",
            mime="text/csv",
            use_container_width=True
        )

    # Explanation of Forecast convergence
    if not np.isnan(uncond_vol_annual):
        st.markdown(f"""
        > **💡 Forecasting Insights:**  
        > The GARCH(1,1) model forecasts conditional volatility by projecting the decay of the current variance shock. Notice how the forecast curves towards the **Long-Run Unconditional Volatility** of **{uncond_vol_annual*100:.2f}%** (green dashed line). 
        > This process is known as **mean reversion** in variance: when current volatility is above the long-run mean, it decreases; when it is below, it increases.
        """)
    else:
        st.markdown("""
        > **💡 Forecasting Insights (Non-Stationary Variance):**  
        > The combined parameters $\\alpha + \\beta$ are equal to or greater than $1.0$ (Persistence $\\ge 100\\%$). This indicates that variance shocks do not decay over time and are permanent. In this state, the long-run unconditional volatility is undefined, and the forecast may not revert to a stable mean.
        """)

# Footer info
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #64748b; font-size: 0.85rem; margin-top: 20px;'>"
    "Developed with Streamlit and statsmodels/arch libraries • GARCH(1,1) Volatility Forecasting Lab"
    "</div>",
    unsafe_allow_html=True
)
