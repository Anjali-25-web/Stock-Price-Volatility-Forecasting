import numpy as np
import pandas as pd
from arch import arch_model

def fit_garch(returns: pd.Series, dist: str = 'normal') -> dict:
    """
    Fits a GARCH(1,1) model to a series of daily log returns.
    Scales returns by 100 for numerical convergence stability as recommended by 'arch'.
    
    Parameters:
        returns (pd.Series): Daily log returns (unscaled).
        dist (str): Distribution of error terms: 'normal', 't' (Student's t), or 'skewt'.
        
    Returns:
        dict: A dictionary containing the fit model, parameters summary, AIC, BIC, 
              and fitted daily/annualized conditional volatility.
    """
    # Scale returns by 100 (crucial for optimizer convergence in arch library)
    scaled_returns = returns * 100
    
    # Specify the GARCH(1,1) model
    # Constant mean model: y_t = mu + epsilon_t
    # GARCH(1,1) volatility model: sigma_t^2 = omega + alpha * epsilon_{t-1}^2 + beta * sigma_{t-1}^2
    model = arch_model(
        scaled_returns, 
        vol='Garch', 
        p=1, 
        q=1, 
        mean='Constant', 
        dist=dist
    )
    
    # Fit the model
    # disp='off' suppresses the optimization solver output to keep the logs clean
    fitted_model = model.fit(disp='off')
    
    # Descale the fitted daily conditional volatility back to the original scale
    fitted_vol_daily = fitted_model.conditional_volatility / 100
    fitted_vol_annualized = fitted_vol_daily * np.sqrt(252)
    
    # Extract model parameters and format as a nice DataFrame
    params_df = pd.DataFrame({
        "Coefficient": fitted_model.params,
        "Std Error": fitted_model.std_err,
        "t-Statistic": fitted_model.tvalues,
        "p-Value": fitted_model.pvalues
    })
    
    # Add a column indicating if parameter is statistically significant at 5% level
    params_df["Significant (5%)"] = params_df["p-Value"] < 0.05
    
    # Calculate GARCH persistence and unconditional volatility
    omega = fitted_model.params.get("omega", 0.0)
    alpha = fitted_model.params.get("alpha[1]", 0.0)
    beta = fitted_model.params.get("beta[1]", 0.0)
    persistence = alpha + beta
    
    if 0 < persistence < 1:
        uncond_var_scaled = omega / (1.0 - persistence)
        uncond_vol_daily = np.sqrt(uncond_var_scaled) / 100
        uncond_vol_annual = uncond_vol_daily * np.sqrt(252)
    else:
        uncond_vol_daily = np.nan
        uncond_vol_annual = np.nan

    return {
        "model_fit": fitted_model,
        "aic": float(fitted_model.aic),
        "bic": float(fitted_model.bic),
        "log_likelihood": float(fitted_model.loglikelihood),
        "parameters": params_df,
        "fitted_volatility_daily": fitted_vol_daily,
        "fitted_volatility_annualized": fitted_vol_annualized,
        "persistence": float(persistence),
        "unconditional_volatility_daily": float(uncond_vol_daily),
        "unconditional_volatility_annualized": float(uncond_vol_annual),
        "scaled": True,
        "original_returns": returns
    }

def forecast_volatility(fit_results: dict, horizon: int = 30) -> pd.DataFrame:
    """
    Forecasts future conditional volatility for a given number of trading days.
    Descales the forecasts back to original returns scale.
    
    Parameters:
        fit_results (dict): The output dictionary from fit_garch.
        horizon (int): Number of trading days to forecast (e.g., 30).
        
    Returns:
        pd.DataFrame: Volatility forecasts indexed by future business dates.
    """
    res = fit_results["model_fit"]
    returns = fit_results["original_returns"]
    
    # Generate multi-step forecast
    # reindex=False prevents warnings for missing indices and speeds up the forecast call
    forecasts = res.forecast(horizon=horizon, reindex=False)
    
    # The variance forecast is a DataFrame where the last row represents the forecast 
    # for the next H periods out of sample.
    last_row_var = forecasts.variance.iloc[-1]
    
    # Extract variance forecasts and calculate standard deviations (volatilities)
    # The variance forecasts are scaled by 100^2 = 10000. So we square root and divide by 100.
    forecasted_vol_daily = np.sqrt(last_row_var) / 100
    forecasted_vol_annualized = forecasted_vol_daily * np.sqrt(252)
    
    # Generate future business days for index
    last_date = returns.index[-1]
    # pd.bdate_range generates business days (excluding weekends)
    forecast_dates = pd.bdate_range(start=last_date + pd.Timedelta(days=1), periods=horizon)
    
    # Construct final dataframe
    forecast_df = pd.DataFrame({
        "Day": np.arange(1, horizon + 1),
        "Daily Volatility (%)": forecasted_vol_daily.values * 100, # Display in percent
        "Annualized Volatility (%)": forecasted_vol_annualized.values * 100, # Display in percent
        "Daily Volatility (Decimal)": forecasted_vol_daily.values,
        "Annualized Volatility (Decimal)": forecasted_vol_annualized.values
    }, index=forecast_dates)
    
    forecast_df.index.name = "Date"
    return forecast_df
