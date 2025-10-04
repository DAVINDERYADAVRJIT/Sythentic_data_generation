import pandas as pd
import numpy as np
from scipy import stats
from typing import Dict, List, Tuple, Any
import warnings
warnings.filterwarnings('ignore')


def find_distributions(
    df: pd.DataFrame,
    max_categories: int = 10,
    alpha: float = 0.05,
    top_n: int = 3
) -> pd.DataFrame:
    """
    Automatically identify the best-fitting probability distribution for each feature.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe with features to analyze
    max_categories : int
        Max unique values to consider a feature categorical (default: 10)
    alpha : float
        Significance level for goodness-of-fit tests (default: 0.05)
    top_n : int
        Number of top candidate distributions to return per feature (default: 3)
    
    Returns:
    --------
    pd.DataFrame with columns:
        - feature: feature name
        - data_type: detected type (binary, categorical, discrete_count, continuous)
        - best_distribution: best fitting distribution
        - parameters: fitted parameters
        - goodness_of_fit: test statistic (lower is better)
        - candidates: alternative distributions ranked by fit
    """
    
    results = []
    
    for col in df.columns:
        result = _analyze_feature(df[col], col, max_categories, alpha, top_n)
        results.append(result)
    
    return pd.DataFrame(results)


def _analyze_feature(
    series: pd.Series,
    col_name: str,
    max_categories: int,
    alpha: float,
    top_n: int
) -> Dict[str, Any]:
    """Analyze a single feature and determine its distribution."""
    
    # Remove NaN values
    data = series.dropna()
    
    if len(data) == 0:
        return {
            'feature': col_name,
            'data_type': 'empty',
            'best_distribution': 'N/A',
            'parameters': {},
            'goodness_of_fit': np.inf,
            'candidates': []
        }
    
    # Detect data type
    data_type, is_discrete = _detect_data_type(data, max_categories)
    
    # Get candidate distributions based on data type
    candidates = _get_candidate_distributions(data, data_type, is_discrete)
    
    # Fit distributions and rank them
    fitted = _fit_and_rank_distributions(data, candidates, is_discrete)
    
    if len(fitted) == 0:
        return {
            'feature': col_name,
            'data_type': data_type,
            'best_distribution': 'Unknown',
            'parameters': {},
            'goodness_of_fit': np.inf,
            'candidates': []
        }
    
    # Get top result
    best = fitted[0]
    top_candidates = [
        f"{d['name']} (fit: {d['goodness_of_fit']:.4f})" 
        for d in fitted[:top_n]
    ]
    
    return {
        'feature': col_name,
        'data_type': data_type,
        'best_distribution': best['name'],
        'parameters': best['params'],
        'goodness_of_fit': best['goodness_of_fit'],
        'candidates': top_candidates
    }


def _detect_data_type(data: pd.Series, max_categories: int) -> Tuple[str, bool]:
    """Detect the type of data: binary, categorical, discrete count, or continuous."""
    
    unique_vals = data.nunique()
    
    # Binary
    if unique_vals == 2:
        return 'binary', True
    
    # Categorical
    if unique_vals <= max_categories and data.dtype == 'object':
        return 'categorical', True
    
    # Check if discrete (integers only)
    is_discrete = np.allclose(data, data.astype(int))
    
    if is_discrete:
        # All non-negative integers suggests count data
        if (data >= 0).all() and unique_vals < len(data) / 2:
            return 'discrete_count', True
        else:
            return 'discrete', True
    
    return 'continuous', False


def _get_candidate_distributions(
    data: pd.Series,
    data_type: str,
    is_discrete: bool
) -> List[Tuple[str, Any]]:
    """Get candidate distributions based on data characteristics."""
    
    candidates = []
    
    if data_type == 'binary':
        candidates = [('bernoulli', stats.bernoulli)]
    
    elif data_type == 'categorical':
        # For categorical, we just note it
        return [('categorical', None)]
    
    elif data_type == 'discrete_count':
        candidates = [
            ('poisson', stats.poisson),
            ('nbinom', stats.nbinom),
            ('geometric', stats.geom),
        ]
        
        # If bounded, add binomial
        if data.max() < 100:
            candidates.append(('binomial', stats.binom))
    
    elif data_type == 'discrete':
        candidates = [
            ('poisson', stats.poisson),
            ('nbinom', stats.nbinom),
        ]
    
    elif data_type == 'continuous':
        # Check data characteristics
        is_positive = (data > 0).all()
        is_bounded_01 = (data >= 0).all() and (data <= 1).all()
        is_skewed = abs(stats.skew(data)) > 1
        
        if is_bounded_01:
            candidates = [
                ('beta', stats.beta),
                ('uniform', stats.uniform),
            ]
        elif is_positive:
            candidates = [
                ('lognorm', stats.lognorm),
                ('gamma', stats.gamma),
                ('expon', stats.expon),
                ('weibull_min', stats.weibull_min),
            ]
            if is_skewed:
                candidates.insert(0, ('lognorm', stats.lognorm))
        else:
            candidates = [
                ('norm', stats.norm),
                ('t', stats.t),
                ('laplace', stats.laplace),
            ]
            # Add lognorm with shift if data has negative values
            if data.min() < 0:
                candidates.append(('norm', stats.norm))
    
    return candidates


def _fit_and_rank_distributions(
    data: np.ndarray,
    candidates: List[Tuple[str, Any]],
    is_discrete: bool
) -> List[Dict[str, Any]]:
    """Fit each candidate distribution and rank by goodness of fit."""
    
    results = []
    
    for name, dist in candidates:
        if dist is None:
            # Categorical - just count frequencies
            results.append({
                'name': name,
                'params': {},
                'goodness_of_fit': 0,
            })
            continue
        
        try:
            # Fit the distribution
            if name == 'bernoulli':
                p = data.mean()
                params = {'p': p}
                fitted_dist = dist(p)
            
            elif name == 'binomial':
                n = int(data.max())
                p = data.mean() / n
                params = {'n': n, 'p': p}
                fitted_dist = dist(n, p)
            
            elif name == 'poisson':
                mu = data.mean()
                params = {'mu': mu}
                fitted_dist = dist(mu)
            
            elif name == 'geometric':
                p = 1 / data.mean() if data.mean() > 0 else 0.5
                p = min(max(p, 0.001), 0.999)
                params = {'p': p}
                fitted_dist = dist(p)
            
            elif name == 'nbinom':
                # Negative binomial: fit using method of moments
                mean_val = data.mean()
                var_val = data.var()
                if var_val > mean_val:
                    p = mean_val / var_val
                    n = mean_val * p / (1 - p)
                    params = {'n': n, 'p': p}
                    fitted_dist = dist(n, p)
                else:
                    continue
            
            else:
                # Continuous distributions - use MLE
                fit_params = dist.fit(data)
                params = dict(zip(['param' + str(i) for i in range(len(fit_params))], fit_params))
                fitted_dist = dist(*fit_params)
            
            # Calculate goodness of fit using KS test or Chi-square
            if is_discrete:
                # Chi-square test for discrete
                unique, counts = np.unique(data, return_counts=True)
                expected = len(data) * fitted_dist.pmf(unique)
                expected = np.maximum(expected, 1)  # Avoid division by zero
                chi2 = np.sum((counts - expected) ** 2 / expected)
                gof = chi2
            else:
                # KS test for continuous
                ks_stat, p_value = stats.kstest(data, fitted_dist.cdf)
                gof = ks_stat
            
            results.append({
                'name': name,
                'params': params,
                'goodness_of_fit': gof,
            })
        
        except Exception as e:
            # Skip distributions that fail to fit
            continue
    
    # Sort by goodness of fit (lower is better)
    results.sort(key=lambda x: x['goodness_of_fit'])
    
    return results


# Example usage
if __name__ == "__main__":
    # Create sample data
    np.random.seed(42)
    
    sample_df = pd.DataFrame({
        'binary_feature': np.random.choice([0, 1], 1000, p=[0.3, 0.7]),
        'count_feature': np.random.poisson(5, 1000),
        'continuous_normal': np.random.normal(100, 15, 1000),
        'continuous_lognormal': np.random.lognormal(3, 0.5, 1000),
        'proportion': np.random.beta(2, 5, 1000),
        'overdispersed_count': np.random.negative_binomial(5, 0.3, 1000),
    })
    
    # Find distributions
    dist_results = find_distributions(sample_df, top_n=3)
    
    # Display results
    print("\n" + "="*80)
    print("AUTOMATIC DISTRIBUTION DETECTION RESULTS")
    print("="*80 + "\n")
    
    for _, row in dist_results.iterrows():
        print(f"Feature: {row['feature']}")
        print(f"  Data Type: {row['data_type']}")
        print(f"  Best Distribution: {row['best_distribution']}")
        print(f"  Parameters: {row['parameters']}")
        print(f"  Goodness of Fit: {row['goodness_of_fit']:.4f}")
        print(f"  Top Candidates: {', '.join(row['candidates'])}")
        print()