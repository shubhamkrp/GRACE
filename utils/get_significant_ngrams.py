import pandas as pd
import numpy as np
from scipy.stats import chi2

def likelihood_ratio_test(trigram, ip_trigrams_dict, op_trigrams_dict, total_IP, total_OP):
    """
    Calculate the log-likelihood ratio for a trigram
    
    Args:
    trigram (str): The trigram to analyze
    ip_trigrams_dict (dict): Trigram counts for ip group
    op_trigrams_dict (dict): Trigram counts for op group
    total_IP (int): Total count in ip group
    total_OP (int): Total count in op group
    
    Returns:
    float: Log-likelihood ratio
    """
    # Get trigram counts
    a = ip_trigrams_dict.get(trigram, 0)  # Count in IP group
    b = op_trigrams_dict.get(trigram, 0)  # Count in OP group
    
    # Total counts
    n11 = a  # Trigram in IP group
    n12 = total_IP - a  # No trigram in IP group
    n21 = b  # Trigram in OP group
    n22 = total_OP - b  # No trigram in OP group
    n = n11 + n12 + n21 + n22  # Total observations
    
    # Compute log-likelihood under null hypothesis (words are independent)
    # Null hypothesis: P(trigram in IP) = P(trigram in OP)
    p11_null = (n11 + n21) / n
    p12_null = (n12 + n22) / n
    
    # Compute log-likelihood under alternative hypothesis
    # Alternative hypothesis: probabilities can differ
    p11_alt = n11 / (n11 + n12) if (n11 + n12) > 0 else 0
    p12_alt = n12 / (n11 + n12) if (n11 + n12) > 0 else 0
    p21_alt = n21 / (n21 + n22) if (n21 + n22) > 0 else 0
    p22_alt = n22 / (n21 + n22) if (n21 + n22) > 0 else 0
    
    # Compute log-likelihood
    def safe_log(x):
        return np.log(x) if x > 0 else 0
    
    # Log-likelihood under null hypothesis
    ll_null = (
        n11 * safe_log(p11_null) + n12 * safe_log(p12_null) +
        n21 * safe_log(p11_null) + n22 * safe_log(p12_null)
    )
    
    # Log-likelihood under alternative hypothesis
    ll_alt = (
        n11 * safe_log(p11_alt) + n12 * safe_log(p12_alt) +
        n21 * safe_log(p21_alt) + n22 * safe_log(p22_alt)
    )
    
    # Likelihood ratio test statistic (2 * log-likelihood difference)
    lr_statistic = 2 * (ll_alt - ll_null)
    
    return lr_statistic

def analyze_trigrams_from_csv(ip_csv_path, op_csv_path, alpha=0.01):
    """
    Analyze trigrams from CSV files using likelihood ratio test
    
    Args:
    ip_csv_path (str): Path to IP trigrams CSV file
    op_csv_path (str): Path to OP trigrams CSV file
    alpha (float): Significance level
    
    Returns:
    tuple: Sets of significant trigrams for each group
    """
    ip_df = pd.read_csv(ip_csv_path)
    op_df = pd.read_csv(op_csv_path)
    
    ip_trigrams_dict = dict(zip(ip_df['Trigram'], ip_df['IP_Count']))
    op_trigrams_dict = dict(zip(op_df['Trigram'], op_df['OP_Count']))
    
    total_IP = ip_df['IP_Count'].sum()
    total_OP = op_df['OP_Count'].sum()
    
    # Get all unique trigrams
    all_trigrams = set(ip_trigrams_dict.keys()).union(set(op_trigrams_dict.keys()))
    
    significant_trigrams = []
    ip_significant_set = set()
    op_significant_set = set()
    
    results_data = []
    
    for trigram in all_trigrams:
        # Calculate likelihood ratio test statistic
        lr_statistic = likelihood_ratio_test(
            trigram, ip_trigrams_dict, op_trigrams_dict, 
            total_IP, total_OP
        )
        
        # Degrees of freedom (for 2x2 contingency table)
        dof = 1
        
        # Get p-value from chi-square distribution
        p_value = 1 - chi2.cdf(lr_statistic, dof)
        
        a = ip_trigrams_dict.get(trigram, 0)
        b = op_trigrams_dict.get(trigram, 0)
        
        # Check statistical significance
        is_significant = p_value < alpha
        
        # Store results
        results_data.append({
            'Trigram': trigram,
            'IP_Count': a,
            'OP_Count': b,
            'Likelihood_Ratio': lr_statistic,
            'P_Value': p_value,
            'Statistically_Significant': is_significant
        })
        
        # If statistically significant, categorize
        if is_significant:
            significant_trigrams.append(trigram)
            
            if a > b:
                ip_significant_set.add(trigram)
            elif b > a:
                op_significant_set.add(trigram)
            else:
                # If equal, add to both sets
                ip_significant_set.add(trigram)
                op_significant_set.add(trigram)
    
    results_df = pd.DataFrame(results_data)
    
    results_df = results_df.sort_values('P_Value')
    
    return significant_trigrams, ip_significant_set, op_significant_set, results_df
