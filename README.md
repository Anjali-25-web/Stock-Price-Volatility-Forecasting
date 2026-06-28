# Stock Price Volatility Forecasting Dashboard

An interactive, end-to-end Python financial modeling dashboard that calculates, visualizes, and forecasts stock price volatility using **GARCH(1,1)** (Generalized Autoregressive Conditional Heteroskedasticity) models. Built with **Streamlit** and **Plotly** for high-performance financial data rendering.

---

## 🚀 Features

- **Historical Stock Data Downloader**: Pulls 5+ years of daily data dynamically from Yahoo Finance (`yfinance`) with local caching to speed up subsequent queries.
- **Exploratory Data Analysis (EDA)**:
  - Interactive price timeline and daily log return plots.
  - Return distribution histogram overlaid with a fitted Gaussian probability density function (PDF) to visually demonstrate **leptokurtosis (fat tails)**.
- **Statistical Diagnostics**:
  - **Augmented Dickey-Fuller (ADF)** stationarity testing with statistic, p-value, and critical values.
  - Interactive rolling historical volatility calculations (daily and annualized).
- **GARCH(1,1) Volatility Engine**:
  - Fits a GARCH(1,1) variance model using the `arch` library.
  - Supports multiple error distributions: Gaussian (Normal), Student's $t$ (to capture fat tails), and Skewed Student's $t$ (to capture asymmetry).
  - Handles numerical scaling and descaling to guarantee optimizer convergence.
- **Volatility Forecasting**:
  - Predicts conditional volatility up to 90 trading days ahead.
  - Visualizes forecast paths alongside the long-run **Unconditional Volatility** to illustrate the financial concept of **mean reversion**.
- **Model Evaluation**: Computes Log-Likelihood, AIC, and BIC metrics.
- **Data Export**: Option to download forecast results as a CSV file.
- **Modern UI/UX**: Professional dark glassmorphic layout styled with custom CSS.

---

## 🧬 Scientific Background: GARCH(1,1)

Stock returns exhibit **volatility clustering**—periods of high volatility are followed by high volatility, and periods of low volatility are followed by low volatility. Standard time-series models assume homoskedasticity (constant variance). GARCH models capture conditional heteroskedasticity (time-varying variance).

The GARCH(1,1) model consists of two equations:

1. **Mean Equation**:
   $$r_t = \mu + \epsilon_t$$
   Where $r_t$ is the daily log return, $\mu$ is the constant mean, and $\epsilon_t$ is the residual shock.

2. **Conditional Variance Equation**:
   $$\sigma_t^2 = \omega + \alpha \epsilon_{t-1}^2 + \beta \sigma_{t-1}^2$$
   Where:
   - $\sigma_t^2$ is the conditional variance on day $t$.
   - $\omega$ is the constant variance term ($\omega > 0$).
   - $\alpha$ is the ARCH coefficient, capturing response to recent shocks ($\alpha \ge 0$).
   - $\beta$ is the GARCH coefficient, capturing memory/persistence of past volatility ($\beta \ge 0$).
   - **Persistence** is defined as $\alpha + \beta$. The process is stationary if $\alpha + \beta < 1$.
   - **Unconditional Volatility** (long-run mean volatility) is defined as:
     $$\sigma_{uncond} = \sqrt{\frac{\omega}{1 - \alpha - \beta}}$$

---

## 📂 Project Structure

```text
Stock_volatility_forecasting/
├── app.py              # Main Streamlit application dashboard
├── model.py            # GARCH modeling, forecasting, and scaling logic
├── utils.py            # Yahoo Finance downloader, returns, and ADF statistics
├── requirements.txt    # Required Python libraries
├── README.md           # Documentation
├── data/               # Local cache for downloaded stock data CSVs
└── assets/             # Folder for dashboard screenshots
```

---

## 🛠️ Installation & Setup

1. **Clone the repository**:
   ```bash
   git clone https://github.com/your-username/Stock_volatility_forecasting.git
   cd Stock_volatility_forecasting
   ```

2. **Create and activate a virtual environment (Recommended)**:
   ```bash
   # Windows
   py -m venv venv
   venv\Scripts\activate

   # macOS/Linux
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the Streamlit app**:
   ```bash
   streamlit run app.py
   ```

5. Open your web browser and navigate to `http://localhost:8501`.

---

## 🖥️ Screen Previews & Walkthrough

*(Screenshots will be uploaded here after deployment)*
- **Exploratory Data Analysis Tab**: Displays historical stock price charts and log returns histograms vs normal distribution fits.
- **Rolling Volatility Tab**: Renders daily return fluctuations and standard rolling volatility curves.
- **GARCH Model Fit Tab**: Displays parameter coefficients, standard errors, p-values, and forecasted conditional volatility.

---

## 📈 Future Enhancements

- **Asymmetric Models**: Support for EGARCH (Exponential GARCH) and GJR-GARCH models to capture the leverage effect (negative shocks increasing volatility more than positive shocks).
- **Value at Risk (VaR)**: Dynamic calculations of VaR and Conditional VaR (Expected Shortfall) using forecasted GARCH volatility.
- **Backtesting**: Volatility forecast evaluation using realized range-based volatility proxies (e.g., Parkinson, Garman-Klass estimators).
