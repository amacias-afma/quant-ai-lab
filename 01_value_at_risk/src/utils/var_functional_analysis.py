import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

def calculate_quantile_loss(returns, var_estimates, alpha=0.05, weight=1, var_priori=0):
    """
    Calculate the quantile loss (pinball loss) for VaR estimates.
    
    Loss = alpha * (y - var) if y >= var
    Loss = (alpha - 1) * (y - var) if y < var
    """
    errors = returns - var_estimates
    loss = np.where(returns >= var_estimates, 
                    alpha * errors, 
                    (alpha - 1) * errors)
    # print('-----------------------------------')
    # print(np.mean(loss))
    # print(weight * (var_priori - var_estimates)**2)
    # print('-----------------------------------')

    return np.mean(loss + weight * (var_priori - var_estimates)**2)

def analyze_var_functional_v2(df, rolling=132, alpha=0.05, z0=1.645, 
                              std_range=(0, 2), mean_range=(-1, 2), n_points=31, weight=1):
    """
    Analyze the quantile loss functional for different actual mean and std values.
    
    The theoretical optimal VaR is: mu - z0 * sigma
    We test combinations to see which minimizes the loss.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with 'log_ret' column
    rolling : int
        Rolling window size for calculating mean and std
    alpha : float
        Significance level (e.g., 0.05 for 95% confidence)
    z0 : float
        Z-score for the given alpha (e.g., 1.645 for alpha=0.05)
    std_range : tuple
        Range for std multipliers (min, max). Default: (0, 2)
    mean_range : tuple
        Range for mean multipliers (min, max). Default: (-1, 2)
    n_points : int
        Number of points to test in each dimension. Default: 31
    
    Returns:
    --------
    tuple : results_df, loss_matrix, actual_std_values, actual_mean_values, 
            optimal_params, theoretical_loss, avg_mean, avg_std
    """
    # Prepare data
    df_toy = df.copy()
    df_toy['std'] = df_toy['log_ret'].rolling(rolling).std()
    df_toy['mean'] = df_toy['log_ret'].rolling(rolling).mean()
    df_toy['log_ret_p1'] = df_toy['log_ret'].shift(-1)
    df_toy = df_toy.dropna()
    
    # Calculate average mean and std to create meaningful ranges
    avg_mean = df_toy['mean'].mean()
    avg_std = df_toy['std'].mean()
    var_priori = avg_mean - z0 * avg_std
    
    print(f"Average rolling mean: {avg_mean:.6f}")
    print(f"Average rolling std: {avg_std:.6f}")
    print(f"Testing std multipliers from {std_range[0]} to {std_range[1]}")
    print(f"Testing mean multipliers from {mean_range[0]} to {mean_range[1]}")
    print(f"Grid size: {n_points} × {n_points} = {n_points**2} points")
    
    # Define grid of multipliers using the provided ranges
    std_multipliers = np.linspace(std_range[0], std_range[1], n_points)
    mean_multipliers = np.linspace(mean_range[0], mean_range[1], n_points)
    
    # Convert to actual values for plotting
    actual_std_values = avg_std * std_multipliers
    actual_mean_values = avg_mean * mean_multipliers
    
    # Store results
    results = []
    loss_matrix = np.zeros((len(std_multipliers), len(mean_multipliers)))
    
    # Calculate loss for each combination
    for i, aux_std in enumerate(std_multipliers):
        for j, aux_mean in enumerate(mean_multipliers):
            # Calculate VaR estimate using the multipliers
            var_estimate = -z0 * df_toy['std'] * aux_std + df_toy['mean'] * aux_mean
            
            # Calculate quantile loss
            loss = calculate_quantile_loss(df_toy['log_ret_p1'].values, 
                                          var_estimate.values, 
                                          alpha, weight, var_priori)
            
            loss_matrix[i, j] = loss
            results.append({
                'aux_std': aux_std,
                'aux_mean': aux_mean,
                'actual_std': avg_std * aux_std,
                'actual_mean': avg_mean * aux_mean,
                'loss': loss
            })
    
    results_df = pd.DataFrame(results)
    
    # Find optimal values
    optimal_idx = results_df['loss'].idxmin()
    optimal_params = results_df.loc[optimal_idx]
    
    # Find loss at theoretical optimal (aux_std=1, aux_mean=1)
    theoretical_idx = results_df[(results_df['aux_std'] == 1.0) & 
                                  (results_df['aux_mean'] == 1.0)].index[0]
    theoretical_loss = results_df.loc[theoretical_idx, 'loss']
    
    return results_df, loss_matrix, actual_std_values, actual_mean_values, optimal_params, theoretical_loss, avg_mean, avg_std

def plot_3d_surface_v2(loss_matrix, std_values, mean_values, optimal_params, theoretical_loss, avg_mean, avg_std):
    """
    Create a 3D surface plot of the loss functional with actual values on axes.
    """
    fig = plt.figure(figsize=(16, 6))
    
    # 3D Surface plot
    ax1 = fig.add_subplot(121, projection='3d')
    
    X, Y = np.meshgrid(mean_values, std_values)
    
    surf = ax1.plot_surface(X, Y, loss_matrix, cmap=cm.viridis, 
                           alpha=0.8, edgecolor='none')
    
    # Mark the optimal point (using actual values)
    optimal_actual_mean = optimal_params['actual_mean']
    optimal_actual_std = optimal_params['actual_std']
    
    ax1.scatter([optimal_actual_mean], 
               [optimal_actual_std], 
               [optimal_params['loss']], 
               color='red', s=100, marker='o', 
               label=f'Optimal: μ×{optimal_params["aux_mean"]:.2f}, σ×{optimal_params["aux_std"]:.2f}')
    
    # Mark the theoretical point (1, 1) using actual values
    theoretical_actual_mean = avg_mean * 1.0
    theoretical_actual_std = avg_std * 1.0
    
    ax1.scatter([theoretical_actual_mean], 
               [theoretical_actual_std], 
               [theoretical_loss], 
               color='yellow', s=100, marker='*', 
               label=f'Theoretical: μ×1.0, σ×1.0')
    
    ax1.set_xlabel('Mean Component (μ × aux_mean)', fontsize=10)
    ax1.set_ylabel('Std Component (σ × aux_std)', fontsize=10)
    ax1.set_zlabel('Quantile Loss', fontsize=10)
    ax1.set_title('3D Surface: Quantile Loss Functional', fontsize=12, fontweight='bold')
    ax1.legend(fontsize=8)
    fig.colorbar(surf, ax=ax1, shrink=0.5)
    
    # 2D Contour plot
    ax2 = fig.add_subplot(122)
    
    contour = ax2.contourf(X, Y, loss_matrix, levels=20, cmap=cm.viridis)
    ax2.contour(X, Y, loss_matrix, levels=10, colors='black', alpha=0.3, linewidths=0.5)
    
    # Mark optimal and theoretical points
    ax2.scatter([optimal_actual_mean], 
               [optimal_actual_std], 
               color='red', s=150, marker='o', edgecolors='white', linewidths=2,
               label=f'Optimal: μ×{optimal_params["aux_mean"]:.2f}, σ×{optimal_params["aux_std"]:.2f}')
    
    ax2.scatter([theoretical_actual_mean], 
               [theoretical_actual_std], 
               color='yellow', s=200, marker='*', edgecolors='white', linewidths=2,
               label=f'Theoretical: μ×1.0, σ×1.0')
    
    ax2.set_xlabel('Mean Component (μ × aux_mean)', fontsize=10)
    ax2.set_ylabel('Std Component (σ × aux_std)', fontsize=10)
    ax2.set_title('2D Contour: Quantile Loss Functional', fontsize=12, fontweight='bold')
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)
    fig.colorbar(contour, ax=ax2)
    
    # Add text box with reference values
    textstr = f'Avg μ: {avg_mean:.6f}\nAvg σ: {avg_std:.6f}'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    ax2.text(0.02, 0.98, textstr, transform=ax2.transAxes, fontsize=9,
            verticalalignment='top', bbox=props)
    
    plt.tight_layout()
    return fig

def analyze_var_functional(df, rolling=132, alpha=0.05, z0=1.645):
    """
    Analyze the quantile loss functional for different aux_std and aux_mean values.
    
    The theoretical optimal VaR is: mu - z0 * sigma
    We test combinations to see which minimizes the loss.
    """
    # Prepare data
    df_toy = df.copy()
    df_toy['std'] = df_toy['log_ret'].rolling(rolling).std()
    df_toy['mean'] = df_toy['log_ret'].rolling(rolling).mean()
    df_toy['log_ret_p1'] = df_toy['log_ret'].shift(-1)
    df_toy = df_toy.dropna()
    
    # Define grid of values to test
    # Include 1.0 for both to test the "perfect world" case
    std_values = np.linspace(0, 2, 21)  # 0 to 2, including 1
    mean_values = np.linspace(-1, 2, 31)  # -1 to 2, including 1
    
    # Store results
    results = []
    loss_matrix = np.zeros((len(std_values), len(mean_values)))
    
    # Calculate loss for each combination
    for i, aux_std in enumerate(std_values):
        for j, aux_mean in enumerate(mean_values):
            # Calculate VaR estimate
            var_estimate = -z0 * df_toy['std'] * aux_std + df_toy['mean'] * aux_mean
            
            # Calculate quantile loss
            loss = calculate_quantile_loss(df_toy['log_ret_p1'].values, 
                                          var_estimate.values, 
                                          alpha)
            
            loss_matrix[i, j] = loss
            results.append({
                'aux_std': aux_std,
                'aux_mean': aux_mean,
                'loss': loss
            })
    
    results_df = pd.DataFrame(results)
    
    # Find optimal values
    optimal_idx = results_df['loss'].idxmin()
    optimal_params = results_df.loc[optimal_idx]
    
    # Find loss at theoretical optimal (aux_std=1, aux_mean=1)
    theoretical_idx = results_df[(results_df['aux_std'] == 1.0) & 
                                  (results_df['aux_mean'] == 1.0)].index[0]
    theoretical_loss = results_df.loc[theoretical_idx, 'loss']
    
    return results_df, loss_matrix, std_values, mean_values, optimal_params, theoretical_loss

def plot_3d_surface(loss_matrix, std_values, mean_values, optimal_params, theoretical_loss):
    """
    Create a 3D surface plot of the loss functional.
    """
    fig = plt.figure(figsize=(16, 6))
    
    # 3D Surface plot
    ax1 = fig.add_subplot(121, projection='3d')
    
    X, Y = np.meshgrid(mean_values, std_values)
    
    surf = ax1.plot_surface(X, Y, loss_matrix, cmap=cm.viridis, 
                           alpha=0.8, edgecolor='none')
    
    # Mark the optimal point
    ax1.scatter([optimal_params['aux_mean']], 
               [optimal_params['aux_std']], 
               [optimal_params['loss']], 
               color='red', s=100, marker='o', 
               label=f'Optimal: ({optimal_params["aux_mean"]:.2f}, {optimal_params["aux_std"]:.2f})')
    
    # Mark the theoretical point (1, 1)
    ax1.scatter([1.0], [1.0], [theoretical_loss], 
               color='yellow', s=100, marker='*', 
               label=f'Theoretical (1.0, 1.0)')
    
    ax1.set_xlabel('aux_mean', fontsize=10)
    ax1.set_ylabel('aux_std', fontsize=10)
    ax1.set_zlabel('Quantile Loss', fontsize=10)
    ax1.set_title('3D Surface: Quantile Loss Functional', fontsize=12, fontweight='bold')
    ax1.legend()
    fig.colorbar(surf, ax=ax1, shrink=0.5)
    
    # 2D Contour plot
    ax2 = fig.add_subplot(122)
    
    contour = ax2.contourf(X, Y, loss_matrix, levels=20, cmap=cm.viridis)
    ax2.contour(X, Y, loss_matrix, levels=10, colors='black', alpha=0.3, linewidths=0.5)
    
    # Mark optimal and theoretical points
    ax2.scatter([optimal_params['aux_mean']], 
               [optimal_params['aux_std']], 
               color='red', s=150, marker='o', edgecolors='white', linewidths=2,
               label=f'Optimal: ({optimal_params["aux_mean"]:.2f}, {optimal_params["aux_std"]:.2f})')
    
    ax2.scatter([1.0], [1.0], 
               color='yellow', s=200, marker='*', edgecolors='white', linewidths=2,
               label=f'Theoretical (1.0, 1.0)')
    
    ax2.set_xlabel('aux_mean', fontsize=10)
    ax2.set_ylabel('aux_std', fontsize=10)
    ax2.set_title('2D Contour: Quantile Loss Functional', fontsize=12, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    fig.colorbar(contour, ax=ax2)
    
    plt.tight_layout()
    return fig

def print_analysis_summary(optimal_params, theoretical_loss, results_df):
    """
    Print summary of the analysis.
    """
    print("=" * 70)
    print("QUANTILE LOSS FUNCTIONAL ANALYSIS")
    print("=" * 70)
    print(f"\nTheoretical Optimal VaR: mu - z0 * sigma")
    print(f"  (corresponds to aux_mean=1.0, aux_std=1.0)")
    print(f"  Loss at theoretical optimal: {theoretical_loss:.6f}")
    print(f"\nEmpirical Optimal Parameters:")
    print(f"  aux_mean: {optimal_params['aux_mean']:.4f}")
    print(f"  aux_std:  {optimal_params['aux_std']:.4f}")
    print(f"  Loss at empirical optimal: {optimal_params['loss']:.6f}")
    print(f"\nDifference from theoretical:")
    print(f"  Δ aux_mean: {optimal_params['aux_mean'] - 1.0:.4f}")
    print(f"  Δ aux_std:  {optimal_params['aux_std'] - 1.0:.4f}")
    print(f"  Δ Loss:     {optimal_params['loss'] - theoretical_loss:.6f}")
    
    # Show top 5 best parameter combinations
    print(f"\nTop 5 Best Parameter Combinations:")
    print("-" * 70)
    top_5 = results_df.nsmallest(5, 'loss')
    for idx, row in top_5.iterrows():
        print(f"  aux_mean={row['aux_mean']:.3f}, aux_std={row['aux_std']:.3f}, loss={row['loss']:.6f}")
    print("=" * 70)

# Example usage:
if __name__ == "__main__":
    # This would be run from your notebook
    # results_df, loss_matrix, std_values, mean_values, optimal_params, theoretical_loss = \
    #     analyze_var_functional(df, rolling=132, alpha=0.05, z0=1.645)
    # 
    # print_analysis_summary(optimal_params, theoretical_loss, results_df)
    # fig = plot_3d_surface(loss_matrix, std_values, mean_values, optimal_params, theoretical_loss)
    # plt.show()
    pass
