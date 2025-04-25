import numpy as np
from scipy import stats
from scipy.signal import welch
from scipy.fft import fft
from statsmodels.tsa.stattools import adfuller, acf, pacf


def extract_time_domain_features(time_series):
    """
    Extract time domain features from a time series
    """
    # Basic statistics
    mean = np.mean(time_series)
    std = np.std(time_series)
    variance = np.var(time_series)
    min_val = np.min(time_series)
    max_val = np.max(time_series)
    range_val = max_val - min_val
    
    # Shape statistics
    skewness = stats.skew(time_series)
    kurtosis = stats.kurtosis(time_series)
    
    # First and second differences (for volatility)
    diff1 = np.diff(time_series)
    diff2 = np.diff(diff1)
    
    mean_diff1 = np.mean(np.abs(diff1))
    std_diff1 = np.std(diff1)
    mean_diff2 = np.mean(np.abs(diff2))
    std_diff2 = np.std(diff2)
    
    # Quantiles
    q25, q50, q75 = np.percentile(time_series, [25, 50, 75])
    iqr = q75 - q25
    
    # Peaks and troughs
    peaks = np.sum((time_series[1:-1] > time_series[:-2]) & (time_series[1:-1] > time_series[2:]))
    troughs = np.sum((time_series[1:-1] < time_series[:-2]) & (time_series[1:-1] < time_series[2:]))
    peak_trough_ratio = peaks / troughs if troughs > 0 else 0
    
    # Return as dictionary
    features = {
        'mean': mean,
        'std': std,
        'variance': variance,
        'min': min_val,
        'max': max_val,
        'range': range_val,
        'skewness': skewness,
        'kurtosis': kurtosis,
        'mean_abs_diff1': mean_diff1,
        'std_diff1': std_diff1,
        'mean_abs_diff2': mean_diff2,
        'std_diff2': std_diff2,
        'q25': q25,
        'q50': q50,
        'q75': q75,
        'iqr': iqr,
        'peaks': peaks,
        'troughs': troughs,
        'peak_trough_ratio': peak_trough_ratio
    }
    
    return features


def extract_frequency_domain_features(time_series, fs=1.0):
    """
    Extract frequency domain features using FFT
    """
    # Apply FFT and get absolute values (magnitude)
    fft_values = np.abs(fft(time_series))
    n = len(time_series)
    freq = np.fft.fftfreq(n, d=1/fs)
    
    # Only consider positive frequencies (first half)
    fft_values = fft_values[:n//2]
    freq = freq[:n//2]
    
    # Power calculation
    power = (fft_values ** 2) / n
    
    # Main frequency features
    max_power = np.max(power)
    max_power_freq = freq[np.argmax(power)]
    mean_power = np.mean(power)
    std_power = np.std(power)
    
    # Spectral entropy
    power_normalized = power / np.sum(power)
    spectral_entropy = -np.sum(power_normalized * np.log2(power_normalized + 1e-10))
    
    # Spectral centroid
    spectral_centroid = np.sum(freq * power) / np.sum(power) if np.sum(power) > 0 else 0
    
    # Power bands (divide frequency range into quarters)
    freq_bands = np.array_split(np.arange(len(freq)), 4)
    power_bands = [np.sum(power[band]) for band in freq_bands]
    band_ratios = [power_bands[i] / np.sum(power) for i in range(len(power_bands))]
    
    # Return as dictionary
    features = {
        'max_power': max_power,
        'max_power_freq': max_power_freq,
        'mean_power': mean_power,
        'std_power': std_power,
        'spectral_entropy': spectral_entropy,
        'spectral_centroid': spectral_centroid,
        'power_band_1': power_bands[0],
        'power_band_2': power_bands[1],
        'power_band_3': power_bands[2],
        'power_band_4': power_bands[3],
        'band_ratio_1': band_ratios[0],
        'band_ratio_2': band_ratios[1],
        'band_ratio_3': band_ratios[2],
        'band_ratio_4': band_ratios[3]
    }
    
    return features


def extract_statistical_features(time_series, lag=10):
    """
    Extract statistical and time series features
    """
    # Autocorrelation at different lags
    acf_values = acf(time_series, nlags=lag, fft=True)
    
    # Stationarity test (ADF)
    try:
        adf_result = adfuller(time_series)
        adf_statistic = adf_result[0]
        adf_pvalue = adf_result[1]
    except:
        adf_statistic = 0
        adf_pvalue = 1
    
    # Calculate returns
    returns = np.diff(time_series) / time_series[:-1]
    
    # Volatility measures
    rolling_std_3 = np.std([time_series[i:i+3] for i in range(len(time_series)-3)])
    rolling_std_5 = np.std([time_series[i:i+5] for i in range(len(time_series)-5)])
    
    # Trend features
    n = len(time_series)
    x = np.arange(n)
    slope, intercept, r_value, p_value, _ = stats.linregress(x, time_series)
    
    # Crossing the mean count
    mean = np.mean(time_series)
    crossing_mean = np.sum(np.diff(time_series > mean) != 0)
    
    features = {
        'acf_1': acf_values[1],
        'acf_2': acf_values[2],
        'acf_3': acf_values[3],
        'acf_4': acf_values[4],
        'acf_5': acf_values[5],
        'adf_statistic': adf_statistic,
        'adf_pvalue': adf_pvalue,
        'mean_return': np.mean(returns),
        'std_return': np.std(returns),
        'skew_return': stats.skew(returns),
        'kurtosis_return': stats.kurtosis(returns),
        'rolling_std_3': rolling_std_3,
        'rolling_std_5': rolling_std_5,
        'trend_slope': slope,
        'trend_intercept': intercept,
        'trend_r_value': r_value,
        'trend_p_value': p_value,
        'crossing_mean': crossing_mean
    }
    
    return features


def extract_all_features(time_series):
    """
    Extract all time series features
    """
    # Clean the time series (remove NaN and inf)
    time_series = np.array(time_series)
    time_series = time_series[~np.isnan(time_series) & ~np.isinf(time_series)]
    
    # Extract features from different domains
    time_features = extract_time_domain_features(time_series)
    freq_features = extract_frequency_domain_features(time_series)
    stat_features = extract_statistical_features(time_series)
    
    # Combine all features
    all_features = {**time_features, **freq_features, **stat_features}
    
    # Convert to array
    feature_names = list(all_features.keys())
    feature_values = np.array([all_features[name] for name in feature_names])
    
    return feature_values, feature_names 