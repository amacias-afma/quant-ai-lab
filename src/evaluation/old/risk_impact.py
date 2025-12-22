# src/evaluation/risk_impact.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class RiskImpactAnalyzer:
    """
    Translates Volatility Models into Business Impact (Value at Risk).
    """
    def __init__(self, returns, vol_model_a, vol_model_b, model_names=("GARCH", "LSTM")):
        """
        :param returns: Series of actual log returns
        :param vol_model_a: Annualized volatility series (Model A)
        :param vol_model_b: Annualized volatility series (Model B)
        """
        self.returns = np.array(returns)
        # De-annualize vol to get Daily Volatility for VaR
        self.vol_a = np.array(vol_model_a) / np.sqrt(252)
        self.vol_b = np.array(vol_model_b) / np.sqrt(252)
        self.names = model_names
        
    def calculate_var_breaches(self, portfolio_value=10_000_000, confidence=0.99):
        """
        Simulates a VaR Backtest.
        VaR (99%) = Portfolio * 2.33 * Daily_Vol
        """
        # Z-score for 99% confidence (assuming normal distribution for simplicity)
        z_score = 2.33 
        
        # Calculate Required Capital (VaR) for each day
        var_a = portfolio_value * z_score * self.vol_a
        var_b = portfolio_value * z_score * self.vol_b
        
        # Calculate Actual PnL
        actual_pnl = portfolio_value * self.returns
        
        # Check Breaches: When Loss > VaR (Note: Loss is negative PnL)
        # We look for days where PnL < -VaR
        breaches_a = actual_pnl < -var_a
        breaches_b = actual_pnl < -var_b
        
        count_a = np.sum(breaches_a)
        count_b = np.sum(breaches_b)
        
        print(f"\n💰 RISK IMPACT ANALYSIS (${portfolio_value:,.0f} Portfolio)")
        print("-" * 60)
        print(f"{self.names[0]} VaR Breaches: {count_a} days ({(count_a/len(self.returns))*100:.2f}%)")
        print(f"{self.names[1]} VaR Breaches: {count_b} days ({(count_b/len(self.returns))*100:.2f}%)")
        
        return actual_pnl, var_a, var_b, breaches_a, breaches_b

    def plot_capital_shield(self):
        """
        Visualizes the 'Capital Shield'.
        Shows where the Actual Loss pierced the VaR armor.
        """
        pnl, var_a, var_b, breach_a, breach_b = self.calculate_var_breaches()
        
        plt.figure(figsize=(14, 7))
        
        # 1. Plot Actual PnL (only negative values matter for risk)
        # We verify if we are filtering properly.
        # Often easier to plot Losses as positive numbers for comparison against VaR limit
        losses = -pnl
        plt.plot(losses, color='gray', alpha=0.4, label='Actual Daily Loss', linewidth=1)
        
        # 2. Plot VaR Limits (The Shield)
        plt.plot(var_a, color='red', linestyle='--', label=f'{self.names[0]} Capital Req', linewidth=1.5)
        plt.plot(var_b, color='blue', linestyle='-', label=f'{self.names[1]} Capital Req', linewidth=1.5)
        
        # 3. Highlight Breaches (Where Loss > VaR)
        # We identify indices where breaches occurred
        # breach_a is a boolean mask.
        
        # Plot red dots for Model A breaches
        plt.scatter(np.where(breach_a)[0], losses[breach_a], 
                    color='red', marker='x', s=100, label=f'{self.names[0]} Breach', zorder=5)
        
        # Plot blue dots for Model B breaches (if any)
        # Often LSTM has fewer breaches in high vol
        plt.scatter(np.where(breach_b)[0], losses[breach_b], 
                    color='blue', marker='o', s=40, label=f'{self.names[1]} Breach', zorder=6)

        plt.title("Capital Adequacy Test: Did the Model Protect the Bank?")
        plt.ylabel("Daily Loss / Capital Required ($)")
        plt.xlabel("Trading Days")
        plt.legend()
        plt.grid(True, alpha=0.2)
        
        # Focus on the 'Loss' area
        plt.ylim(0, np.max(losses)*1.1) 
        plt.show()