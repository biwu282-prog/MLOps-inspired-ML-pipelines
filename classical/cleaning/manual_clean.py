import pandas as pd
import numpy as np
import re
import os
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import IsolationForest
from scipy.stats import zscore, skew, kurtosis
import matplotlib.pyplot as plt
import argparse

# ========================================
# ===== Path Configuration =====
# ========================================

def get_base_output_dir():
    """
    Get base output directory (Result folder in parent directory of auto6.py)
    
    Returns:
    --------
    str : Base output directory path
    """
    # Get the directory where auto6.py is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Create Result folder in same directory as script
    base_output_dir = os.path.join(script_dir, 'Result')
    
    return base_output_dir


# Define output subdirectories
def get_output_paths():
    """
    Get all output paths
    
    Returns:
    --------
    dict : Dictionary containing all output paths
    """
    base_dir = get_base_output_dir()
    
    return {
        'base': base_dir,
        'clean_data': os.path.join(base_dir, 'Clean_data'),
        'figures': os.path.join(base_dir, 'Figure'),
        'reports': os.path.join(base_dir, 'Clean_data')  # Reports saved with cleaned data
    }


# ========================================
# ===== Method Registration System =====
# ========================================

DETECTION_METHODS = {}
FIXING_METHODS = {}
SMOOTHING_METHODS = {}


def register_detection(name):
    """Register outlier detection method"""
    def decorator(func):
        DETECTION_METHODS[name] = func
        func.method_name = name
        return func
    return decorator


def register_fixing(name):
    """Register outlier fixing method"""
    def decorator(func):
        FIXING_METHODS[name] = func
        func.method_name = name
        return func
    return decorator


def register_smoothing(name):
    """Register data smoothing method"""
    def decorator(func):
        SMOOTHING_METHODS[name] = func
        func.method_name = name
        return func
    return decorator


# ========================================
# ===== Outlier Detection Methods =====
# ========================================

@register_detection('trend_z')
def detect_trend_z(df, col, year_col, z_thresh=2.8, **kwargs):
    """
    Trend-based Z-score detection
    
    Parameters:
    -----------
    z_thresh : float
        Z-score threshold, default 2.8
    """
    X = df[year_col].values.reshape(-1, 1)
    y = df[col].values
    lr = LinearRegression().fit(X, y)
    trend_pred = lr.predict(X)
    resid = y - trend_pred
    z = (resid - resid.mean()) / (resid.std(ddof=0) + 1e-10)
    df["Outlier"] = (np.abs(z) > z_thresh).astype(int)
    df[f"{col}_TrendPred"] = trend_pred
    detect_note = f"Trend-Z (z>{z_thresh})"
    return df, detect_note


@register_detection('modified_z')
def detect_modified_z(df, col, year_col=None, mad_thresh=3.5, **kwargs):
    """
    Modified Z-score based on MAD (Median Absolute Deviation)
    
    Parameters:
    -----------
    mad_thresh : float
        MAD threshold, default 3.5
    """
    median = df[col].median()
    mad = np.median(np.abs(df[col] - median))
    if mad == 0:
        modz = np.zeros(len(df))
    else:
        modz = 0.6745 * (df[col] - median) / mad
    df["Outlier"] = (np.abs(modz) > mad_thresh).astype(int)
    detect_note = f"Modified-Z (MAD>{mad_thresh})"
    return df, detect_note


@register_detection('isolation')
def detect_isolation_forest(df, col, year_col=None, contamination=0.1, **kwargs):
    """
    Isolation Forest outlier detection
    
    Parameters:
    -----------
    contamination : float
        Expected proportion of outliers, default 0.1
    """
    iso = IsolationForest(contamination=contamination, random_state=42)
    preds = iso.fit_predict(df[[col]].values.reshape(-1, 1))
    df["Outlier"] = (preds == -1).astype(int)
    detect_note = f"IsolationForest (contamination={contamination})"
    return df, detect_note


@register_detection('iqr')
def detect_iqr(df, col, year_col=None, iqr_multiplier=1.5, **kwargs):
    """
    Interquartile Range (IQR) method
    
    Parameters:
    -----------
    iqr_multiplier : float
        IQR multiplier, default 1.5
    """
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - iqr_multiplier * IQR
    upper_bound = Q3 + iqr_multiplier * IQR
    df["Outlier"] = ((df[col] < lower_bound) | (df[col] > upper_bound)).astype(int)
    detect_note = f"IQR (multiplier={iqr_multiplier})"
    return df, detect_note


@register_detection('zscore')
def detect_zscore(df, col, year_col=None, z_thresh=3.0, **kwargs):
    """
    Standard Z-score detection
    
    Parameters:
    -----------
    z_thresh : float
        Z-score threshold, default 3.0
    """
    z = np.abs(zscore(df[col]))
    df["Outlier"] = (z > z_thresh).astype(int)
    detect_note = f"Z-score (z>{z_thresh})"
    return df, detect_note


# ========================================
# ===== Outlier Fixing Methods =====
# ========================================

@register_fixing('median')
def fix_with_median(df, col, **kwargs):
    """Replace outliers with median value"""
    normal_data = df.loc[df["Outlier"] == 0, col]
    if len(normal_data) > 0:
        median_val = normal_data.median()
        df[col] = df[col].astype(float)
        df.loc[df["Outlier"] == 1, col] = median_val
    return df


@register_fixing('mean')
def fix_with_mean(df, col, **kwargs):
    """Replace outliers with mean value"""
    normal_data = df.loc[df["Outlier"] == 0, col]
    if len(normal_data) > 0:
        mean_val = normal_data.mean()
        df[col] = df[col].astype(float)
        df.loc[df["Outlier"] == 1, col] = mean_val
    return df


@register_fixing('interpolate')
def fix_with_interpolate(df, col, **kwargs):
    """Replace outliers using linear interpolation"""
    df_copy = df.copy()
    df_copy.loc[df_copy["Outlier"] == 1, col] = np.nan
    df_copy[col] = df_copy[col].interpolate(method='linear', limit_direction='both')
    df[col] = df_copy[col]
    return df


@register_fixing('remove')
def fix_with_remove(df, col, **kwargs):
    """Remove outliers (mark as NaN)"""
    df.loc[df["Outlier"] == 1, col] = np.nan
    return df


@register_fixing('clip')
def fix_with_clip(df, col, **kwargs):
    """Clip outliers to normal range boundaries"""
    normal_data = df.loc[df["Outlier"] == 0, col]
    if len(normal_data) > 0:
        lower = normal_data.min()
        upper = normal_data.max()
        df[col] = df[col].astype(float)
        df.loc[df["Outlier"] == 1, col] = df.loc[df["Outlier"] == 1, col].clip(lower, upper)
    return df


# ========================================
# ===== Data Smoothing Methods =====
# ========================================

@register_smoothing('ewma')
def smooth_ewma(series, alpha=0.3, **kwargs):
    """
    Exponentially Weighted Moving Average
    
    Parameters:
    -----------
    alpha : float
        Smoothing coefficient, default 0.3
    """
    return series.ewm(alpha=alpha, adjust=False).mean()


@register_smoothing('rolling')
def smooth_rolling(series, window=3, **kwargs):
    """
    Rolling window average
    
    Parameters:
    -----------
    window : int
        Window size, default 3
    """
    return series.rolling(window=window, min_periods=1, center=True).mean()


@register_smoothing('savgol')
def smooth_savgol(series, window=5, polyorder=2, **kwargs):
    """
    Savitzky-Golay filter
    
    Parameters:
    -----------
    window : int
        Window size, default 5 (must be odd)
    polyorder : int
        Polynomial order, default 2
    """
    from scipy.signal import savgol_filter
    # Ensure window is odd
    if window % 2 == 0:
        window += 1
    # Ensure window > polyorder
    if window <= polyorder:
        window = polyorder + 2
    try:
        smoothed = savgol_filter(series, window, polyorder)
        return pd.Series(smoothed, index=series.index)
    except:
        # Fallback to simple smoothing if failed
        return series.rolling(window=3, min_periods=1, center=True).mean()


@register_smoothing('gaussian')
def smooth_gaussian(series, sigma=1.0, **kwargs):
    """
    Gaussian filter
    
    Parameters:
    -----------
    sigma : float
        Standard deviation of Gaussian kernel, default 1.0
    """
    from scipy.ndimage import gaussian_filter1d
    smoothed = gaussian_filter1d(series.values, sigma=sigma)
    return pd.Series(smoothed, index=series.index)


@register_smoothing('none')
def smooth_none(series, **kwargs):
    """No smoothing applied"""
    return series

# ========================================
# ===== Missing Value Handling (EWMA) =====
# ========================================

def record_missing_values(df, year_col=None):
    """
    Record detailed information about missing values
    
    Parameters:
    -----------
    df : DataFrame
        Input dataframe
    year_col : str
        Year column name (optional)
    
    Returns:
    --------
    dict : Detailed missing value statistics
    """
    missing_info = {
        'total_missing': 0,
        'columns': {},
        'positions': {}
    }
    
    for col in df.columns:
        if col == year_col:
            continue
            
        if np.issubdtype(df[col].dtype, np.number):
            missing_mask = df[col].isnull()
            missing_count = missing_mask.sum()
            
            if missing_count > 0:
                missing_info['total_missing'] += missing_count
                
                # Record column-level statistics
                missing_info['columns'][col] = {
                    'count': missing_count,
                    'percentage': (missing_count / len(df) * 100),
                    'indices': df[missing_mask].index.tolist()
                }
                
                # Record positions with year information
                if year_col and year_col in df.columns:
                    missing_years = df.loc[missing_mask, year_col].tolist()
                    missing_info['positions'][col] = [
                        {'row': idx, 'year': year} 
                        for idx, year in zip(df[missing_mask].index.tolist(), missing_years)
                    ]
                else:
                    missing_info['positions'][col] = [
                        {'row': idx} 
                        for idx in df[missing_mask].index.tolist()
                    ]
    
    return missing_info


def handle_missing_with_ewma(series, alpha=0.3):
    """
    Fill missing values using EWMA (Exponentially Weighted Moving Average)
    
    Parameters:
    -----------
    series : Series
        Input series with potential missing values
    alpha : float
        Smoothing coefficient for EWMA, default 0.3
    
    Returns:
    --------
    Series : Series with missing values filled
    
    Strategy:
    ---------
    1. Record original missing positions
    2. Forward fill to establish baseline
    3. Apply EWMA smoothing
    4. Backward fill any remaining NaN at start
    5. Replace only originally missing values
    """
    missing_mask = series.isnull()
    
    if missing_mask.sum() == 0:
        return series
    
    # Create working copy
    filled = series.copy()
    
    # Forward fill first
    filled = filled.fillna(method='ffill')
    
    # Backward fill for any NaN at beginning
    filled = filled.fillna(method='bfill')
    
    # Apply EWMA smoothing
    ewma_smoothed = filled.ewm(alpha=alpha, adjust=False).mean()
    
    # Replace only originally missing values
    result = series.copy()
    result[missing_mask] = ewma_smoothed[missing_mask]
    
    return result


def handle_all_missing_values(df, year_col=None, alpha=0.3):
    """
    Handle missing values in all numeric columns using EWMA
    
    Parameters:
    -----------
    df : DataFrame
        Input dataframe
    year_col : str
        Year column name (optional)
    alpha : float
        EWMA alpha parameter (default: 0.3)
    
    Returns:
    --------
    DataFrame : Dataframe with missing values handled
    dict : Statistics about missing values (before and after)
    """
    # Record BEFORE handling
    missing_before = record_missing_values(df, year_col)
    
    if missing_before['total_missing'] == 0:
        return df, None
    
    # Handle missing values
    df_cleaned = df.copy()
    
    for col in df_cleaned.columns:
        if col == year_col:
            continue
            
        if np.issubdtype(df_cleaned[col].dtype, np.number):
            if df_cleaned[col].isnull().sum() > 0:
                df_cleaned[col] = handle_missing_with_ewma(df_cleaned[col], alpha=alpha)
    
    # Record AFTER handling
    missing_after = record_missing_values(df_cleaned, year_col)
    
    # Combine statistics
    missing_stats = {
        'method': 'ewma',
        'alpha': alpha,
        'before': missing_before,
        'after': missing_after,
        'filled_count': missing_before['total_missing'] - missing_after['total_missing']
    }
    
    return df_cleaned, missing_stats

# ========================================
# ===== Unified Interfaces =====
# ========================================

def detect_outliers(df, col, year_col, method="trend_z", **kwargs):
    """
    Detect outliers
    
    Parameters:
    -----------
    df : DataFrame
        Input dataframe
    col : str
        Column name to detect
    year_col : str
        Year column name
    method : str
        Detection method name
    **kwargs : dict
        Parameters passed to detection method
    """
    if method not in DETECTION_METHODS:
        available = ', '.join(sorted(DETECTION_METHODS.keys()))
        raise ValueError(
            f"[ERROR] Unknown detection method: '{method}'\n"
            f"Available methods: {available}"
        )
    
    detection_func = DETECTION_METHODS[method]
    return detection_func(df.copy(), col, year_col, **kwargs)


def fix_outliers(df, col, method="median", **kwargs):
    """
    Fix outliers
    
    Parameters:
    -----------
    df : DataFrame
        Input dataframe
    col : str
        Column name to fix
    method : str
        Fixing method name
    **kwargs : dict
        Parameters passed to fixing method
    """
    if method not in FIXING_METHODS:
        available = ', '.join(sorted(FIXING_METHODS.keys()))
        raise ValueError(
            f"[ERROR] Unknown fixing method: '{method}'\n"
            f"Available methods: {available}"
        )
    
    fixing_func = FIXING_METHODS[method]
    return fixing_func(df.copy(), col, **kwargs)


def smooth_series(series, method="ewma", **kwargs):
    """
    Smooth data series
    
    Parameters:
    -----------
    series : Series
        Input series to smooth
    method : str
        Smoothing method name
    **kwargs : dict
        Parameters passed to smoothing method
    """
    if method not in SMOOTHING_METHODS:
        available = ', '.join(sorted(SMOOTHING_METHODS.keys()))
        raise ValueError(
            f"[ERROR] Unknown smoothing method: '{method}'\n"
            f"Available methods: {available}"
        )
    
    smoothing_func = SMOOTHING_METHODS[method]
    return smoothing_func(series.copy(), **kwargs)


def get_available_methods():
    """Get all available methods"""
    return {
        'detection': sorted(DETECTION_METHODS.keys()),
        'fixing': sorted(FIXING_METHODS.keys()),
        'smoothing': sorted(SMOOTHING_METHODS.keys())
    }


# ========================================
# ===== Data Loading & Selection =====
# ========================================

def load_and_select_data(file_path):
    """Automatically detect and extract valid columns (year + numeric)"""
    if file_path.endswith('.csv'):
        df = pd.read_csv(file_path)
    elif file_path.endswith('.xlsx'):
        df = pd.read_excel(file_path, engine='openpyxl')
    elif file_path.endswith('.xls'):
        df = pd.read_excel(file_path, engine='xlrd')
    else:
        raise ValueError(f"Unsupported file format: {file_path}. Please use .csv, .xlsx, or .xls")
    print(f"[OK] Loaded file: {file_path}")
    print(f"  Data dimensions: {df.shape[0]} x {df.shape[1]}")

    # Detect year columns
    year_cols = [col for col in df.columns if re.search("year", col, re.IGNORECASE)]
    
    year_col = None
    if year_cols:
        year_col = year_cols[0]
        if len(year_cols) > 1:
            print(f"  Found multiple year columns: {year_cols}")
            print(f"  Using '{year_col}' as primary year column")
    else:
        # Try to detect numeric year column
        for col in df.columns:
            if np.issubdtype(df[col].dtype, np.number):
                if df[col].between(1900, 2100).sum() > len(df) * 0.6:
                    year_col = col
                    print(f"  Detected numeric year column: {year_col}")
                    break
    
    if not year_col:
        raise ValueError("[ERROR] No year column detected")

    # Detect valid numeric columns
    numeric_cols = []
    for col in df.columns:
        if np.issubdtype(df[col].dtype, np.number) and df[col].nunique() > 3:
            numeric_cols.append(col)

    if not numeric_cols:
        raise ValueError("[ERROR] No valid numeric columns detected")

    # Ensure only one year column is included
    selected_cols = [year_col] + [c for c in numeric_cols if c not in year_cols]
    df_selected = df[selected_cols].copy()

    print(f"  Selected columns: {selected_cols}")

    # Clean data
    df_selected = df_selected.dropna(how="all").drop_duplicates()
    
    # ===== Check and handle missing values (BEFORE outlier detection) =====
    total_missing = df_selected.isnull().sum().sum()

    if total_missing > 0:
        print(f"\n{'='*60}")
        print(f"Missing Value Analysis")
        print(f"{'='*60}")
        print(f"Total missing values: {total_missing}")
        
        for col in df_selected.columns:
            if col != year_col:
                missing_count = df_selected[col].isnull().sum()
                if missing_count > 0:
                    missing_pct = (missing_count / len(df_selected) * 100)
                    print(f"\n{col}:")
                    print(f"  Count: {missing_count} ({missing_pct:.2f}%)")
                    
                    # Show positions
                    missing_mask = df_selected[col].isnull()
                    if year_col in df_selected.columns:
                        missing_years = df_selected.loc[missing_mask, year_col].tolist()
                        if len(missing_years) <= 10:
                            print(f"  Years: {missing_years}")
                        else:
                            print(f"  Years: {missing_years[:5]} ... and {len(missing_years)-5} more")
                    else:
                        missing_rows = df_selected[missing_mask].index.tolist()
                        if len(missing_rows) <= 10:
                            print(f"  Rows: {missing_rows}")
                        else:
                            print(f"  Rows: {missing_rows[:5]} ... and {len(missing_rows)-5} more")
        
        print(f"{'='*60}")
        print(f"Handling missing values using EWMA (alpha=0.3)...")
        print(f"{'='*60}\n")
        
        # Handle missing values with EWMA
        df_selected, missing_stats = handle_all_missing_values(df_selected, year_col=year_col, alpha=0.3)
        
        print(f"[OK] Filled {missing_stats['filled_count']} missing values using EWMA\n")
        
        # Check if any missing values remain
        remaining = missing_stats['after']['total_missing']
        if remaining > 0:
            print(f"[WARNING] {remaining} missing values still remain\n")
    else:
        missing_stats = None
        print(f"  [OK] No missing values found\n")

    return df_selected, year_col, missing_stats


# ========================================
# ===== Visualization =====
# ========================================

def plot_all(df, df_detected, df_fixed, df_smoothed, col, year_col,
             detect_method_note, fix_method_note, file_path, smooth_method_note):
    """Visualize complete data processing pipeline"""
    plt.figure(figsize=(14, 4))

    # (1) Original data
    plt.subplot(1, 3, 1)
    plt.plot(df[year_col], df[col], marker="o", color="steelblue", linewidth=2)
    plt.title(f"(1) Original - {col}", fontsize=12, fontweight='bold')
    plt.xlabel("Year")
    plt.ylabel(col)
    plt.grid(alpha=0.3)

    # (2) Outlier detection
    plt.subplot(1, 3, 2)
    plt.plot(df_detected[year_col], df_detected[col], "b-o", label="Data", linewidth=2)

    # If trend prediction exists, plot trend line
    if f"{col}_TrendPred" in df_detected.columns:
        plt.plot(
            df_detected[year_col],
            df_detected[f"{col}_TrendPred"],
            "--",
            color="orange",
            linewidth=2,
            label="Trend"
        )

    # Mark outliers
    outliers = df_detected[df_detected["Outlier"] == 1]
    if len(outliers) > 0:
        plt.scatter(
            outliers[year_col],
            outliers[col],
            color="r",
            s=100,
            marker='X',
            label="Outlier",
            zorder=5
        )
    
    plt.title(f"(2) Detection ({detect_method_note})", fontsize=12, fontweight='bold')
    plt.xlabel("Year")
    plt.ylabel(col)
    plt.legend()
    plt.grid(alpha=0.3)

    # (3) Fixed and smoothed
    plt.subplot(1, 3, 3)
    plt.plot(df_fixed[year_col], df_fixed[col], "g-o", label=f"Fixed ({fix_method_note})", linewidth=2)
    plt.plot(df_smoothed[year_col], df_smoothed[col], "r--", label=f"Smoothed ({smooth_method_note})", linewidth=2.5)
    plt.title(f"(3) Fixed & Smoothed", fontsize=12, fontweight='bold')
    plt.xlabel("Year")
    plt.ylabel(col)
    plt.legend()
    plt.grid(alpha=0.3)

    plt.tight_layout()

    # Get the base name of the input file (without extension)
    data_filename = os.path.splitext(os.path.basename(file_path))[0]
    
    # Create output directory: Result\Figure\data_filename\
    output_paths = get_output_paths()
    output_dir = os.path.join(output_paths['figures'], data_filename)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # save figure
    save_path = os.path.join(output_dir, f"{col}_processing.png")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f" Saved figure: {save_path}")
    plt.close()

# ========================================
# ===== Report Generation =====
# ========================================

def generate_markdown_report(file_path, args, processing_stats, output_dir):
    """
    Generate processing report in Markdown format
    
    Parameters:
    -----------
    file_path : str
        Input file path
    args : argparse.Namespace
        Command line arguments
    processing_stats : dict
        Processing statistics
    output_dir : str
        Output directory
    """
    from datetime import datetime
    
    report_filename = f"{os.path.splitext(os.path.basename(file_path))[0]}_report.md"
    report_path = os.path.join(output_dir, report_filename)
    
    with open(report_path, 'w', encoding='utf-8') as f:
        # Header
        f.write(f"# Data Processing Report\n\n")
        f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"---\n\n")
        
        # Input Information
        f.write(f"## Input Information\n\n")
        f.write(f"| Item | Value |\n")
        f.write(f"|------|-------|\n")
        f.write(f"| **Input File** | `{os.path.basename(file_path)}` |\n")
        f.write(f"| **File Path** | `{file_path}` |\n")
        f.write(f"| **File Size** | {os.path.getsize(file_path) / 1024:.2f} KB |\n")
        f.write(f"| **Data Rows** | {processing_stats['total_rows']} |\n")
        f.write(f"| **Data Columns** | {processing_stats['total_columns']} |\n")
        f.write(f"| **Year Range** | {processing_stats['year_range']} |\n")
        f.write(f"\n")
        
        # Processing Methods
        f.write(f"## Processing Methods\n\n")
        f.write(f"| Processing Step | Method | Parameters |\n")
        f.write(f"|----------------|--------|------------|\n")
        f.write(f"| **Outlier Detection** | `{args.outlier_detection}` | ")
        
        # Add method-specific parameters
        if args.outlier_detection == 'trend_z':
            f.write(f"z_threshold={args.z_threshold} |\n")
        elif args.outlier_detection == 'isolation':
            f.write(f"contamination={args.contamination} |\n")
        elif args.outlier_detection == 'iqr':
            f.write(f"iqr_multiplier={args.iqr_multiplier} |\n")
        elif args.outlier_detection == 'modified_z':
            f.write(f"mad_threshold={args.mad_threshold} |\n")
        elif args.outlier_detection == 'zscore':
            f.write(f"z_threshold={args.z_threshold} |\n")
        
        f.write(f"| **Outlier Fixing** | `{args.fix_method}` | - |\n")
        
        f.write(f"| **Data Smoothing** | `{args.smooth_method}` | ")
        if args.smooth_method == 'ewma':
            f.write(f"alpha={args.alpha} |\n")
        elif args.smooth_method == 'rolling':
            f.write(f"window={args.window} |\n")
        elif args.smooth_method == 'savgol':
            f.write(f"window={args.window}, polyorder={args.polyorder} |\n")
        elif args.smooth_method == 'gaussian':
            f.write(f"sigma={args.sigma} |\n")
        else:
            f.write(f"- |\n")
        
        f.write(f"\n")
        
        # Processing Results
        f.write(f"## Processing Results\n\n")
        f.write(f"### Summary Statistics\n\n")
        f.write(f"| Metric | Value |\n")
        f.write(f"|--------|-------|\n")
        f.write(f"| **Total Columns Processed** | {processing_stats['columns_processed']} |\n")
        f.write(f"| **Total Outliers Detected** | {processing_stats['total_outliers']} |\n")
        f.write(f"| **Average Outliers per Column** | {processing_stats['avg_outliers']:.2f} |\n")
        f.write(f"| **Outlier Rate** | {processing_stats['outlier_rate']:.2f}% |\n")
        f.write(f"\n")
        
        # Column-wise Results
        f.write(f"### Column-wise Results\n\n")
        f.write(f"| Column | Outliers Detected | Outlier Rate | Mean (Original) | Mean (Cleaned) | Change |\n")
        f.write(f"|--------|-------------------|--------------|-----------------|----------------|--------|\n")
        
        for col_stat in processing_stats['column_stats']:
            mean_change = col_stat['mean_cleaned'] - col_stat['mean_original']
            mean_change_pct = (mean_change / col_stat['mean_original'] * 100) if col_stat['mean_original'] != 0 else 0
            
            f.write(f"| **{col_stat['name']}** | ")
            f.write(f"{col_stat['outliers']} | ")
            f.write(f"{col_stat['outlier_rate']:.2f}% | ")
            f.write(f"{col_stat['mean_original']:.2f} | ")
            f.write(f"{col_stat['mean_cleaned']:.2f} | ")
            f.write(f"{mean_change:+.2f} ({mean_change_pct:+.2f}%) |\n")
        
        f.write(f"\n")
        
        # Data Quality Assessment
        f.write(f"## Data Quality Assessment\n\n")
        f.write(f"### Before Processing\n\n")
        f.write(f"| Column | Min | Max | Mean | Std Dev | Missing |\n")
        f.write(f"|--------|-----|-----|------|---------|----------|\n")
        
        for col_stat in processing_stats['column_stats']:
            f.write(f"| **{col_stat['name']}** | ")
            f.write(f"{col_stat['min_original']:.2f} | ")
            f.write(f"{col_stat['max_original']:.2f} | ")
            f.write(f"{col_stat['mean_original']:.2f} | ")
            f.write(f"{col_stat['std_original']:.2f} | ")
            f.write(f"{col_stat['missing']} |\n")
        
        f.write(f"\n")
        
        f.write(f"### After Processing\n\n")
        f.write(f"| Column | Min | Max | Mean | Std Dev |\n")
        f.write(f"|--------|-----|-----|------|---------|\n")
        
        for col_stat in processing_stats['column_stats']:
            f.write(f"| **{col_stat['name']}** | ")
            f.write(f"{col_stat['min_cleaned']:.2f} | ")
            f.write(f"{col_stat['max_cleaned']:.2f} | ")
            f.write(f"{col_stat['mean_cleaned']:.2f} | ")
            f.write(f"{col_stat['std_cleaned']:.2f} |\n")
        
        f.write(f"\n")
        
        # Output Files
        f.write(f"## Output Files\n\n")
        f.write(f"| File Type | File Name | Location |\n")
        f.write(f"|-----------|-----------|----------|\n")
        f.write(f"| **Cleaned Data** | `{processing_stats['output_filename']}` | `{output_dir}` |\n")
        f.write(f"| **Report (This File)** | `{report_filename}` | `{output_dir}` |\n")
        
        if not args.no_plot:
            f.write(f"| **Visualizations** | `*_processing.png` | `{processing_stats['figure_dir']}` |\n")
        
        f.write(f"\n")
        
        # Footer
        f.write(f"---\n\n")
        f.write(f"*Generated by Auto5 Enhanced Data Processing Tool*\n")
    
    return report_path


def generate_txt_report(file_path, args, processing_stats, output_dir):
    """
    Generate processing report in TXT format
    """
    from datetime import datetime

    report_filename = f"{os.path.splitext(os.path.basename(file_path))[0]}_report.txt"
    report_path = os.path.join(output_dir, report_filename)

    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("=" * 60 + "\n")
        f.write("DATA PROCESSING REPORT\n")
        f.write("=" * 60 + "\n\n")

        f.write("[Input Information]\n")
        f.write(f"  Input File    : {os.path.basename(file_path)}\n")
        f.write(f"  File Path     : {file_path}\n")
        f.write(f"  File Size     : {os.path.getsize(file_path) / 1024:.2f} KB\n")
        f.write(f"  Generated     : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"  Data Rows     : {processing_stats['total_rows']}\n")
        f.write(f"  Data Columns  : {processing_stats['total_columns']}\n")
        f.write(f"  Year Range    : {processing_stats['year_range']}\n\n")

        f.write("[Processing Methods]\n")
        f.write(f"  Outlier Detection : {args.outlier_detection}")
        f.write(f"  ({processing_stats.get('detection_params', '-')})\n")
        f.write(f"  Outlier Fixing    : {args.fix_method}\n")
        f.write(f"  Data Smoothing    : {args.smooth_method}")
        f.write(f"  ({processing_stats.get('smoothing_params', '-')})\n\n")

        f.write("[Summary]\n")
        f.write(f"  Columns Processed    : {processing_stats['columns_processed']}\n")
        f.write(f"  Total Outliers       : {processing_stats['total_outliers']}\n")
        f.write(f"  Avg Outliers/Column  : {processing_stats['avg_outliers']:.2f}\n")
        f.write(f"  Overall Outlier Rate : {processing_stats['outlier_rate']:.2f}%\n\n")

        f.write("[Column Results]\n")
        f.write(f"  {'Column':<25} {'Outliers':>8} {'Rate':>7}  {'Mean(Orig)':>12} {'Mean(Clean)':>12} {'Change':>10}\n")
        f.write("  " + "-" * 78 + "\n")
        for col_stat in processing_stats['column_stats']:
            mean_change = col_stat['mean_cleaned'] - col_stat['mean_original']
            mean_change_pct = (mean_change / col_stat['mean_original'] * 100) if col_stat['mean_original'] != 0 else 0
            f.write(f"  {col_stat['name']:<25} {col_stat['outliers']:>8} {col_stat['outlier_rate']:>6.1f}%"
                    f"  {col_stat['mean_original']:>12.2f} {col_stat['mean_cleaned']:>12.2f}"
                    f"  {mean_change_pct:>+9.2f}%\n")

        f.write("\n[Output Files]\n")
        f.write(f"  Cleaned Data : {processing_stats['output_filename']}\n")
        f.write(f"  TXT Report   : {report_filename}\n")
        if not args.no_plot:
            f.write(f"  Figures      : {processing_stats['figure_dir']}\n")

        f.write("\n" + "=" * 60 + "\n")

    return report_path

# ========================================
# ===== Main Processing Pipeline =====
# ========================================

def auto6_pipeline(file_path, args):
    """
    Main processing pipeline
    
    Parameters:
    -----------
    file_path : str
        Input file path
    args : argparse.Namespace
        Command line arguments
    """
    if args.verbose:
        print(f"\n{'='*60}")
        print(f"DataProcess - Data Processing Tool")
        print(f"{'='*60}")
        print(f"Configuration:")
        print(f"  Input file: {file_path}")
        print(f"  Missing value handling: EWMA (alpha=0.3)")
        print(f"  Outlier detection: {args.outlier_detection}")
        print(f"  Outlier fixing: {args.fix_method}")
        print(f"  Data smoothing: {args.smooth_method}")
        print(f"{'='*60}\n")
    
    # Load data
    df, year_col, missing_stats = load_and_select_data(file_path)
    
    if args.verbose:
        print(f"\n{'='*60}")
        print(f"Data Quality Overview")
        print(f"{'='*60}")
        for col in df.columns:
            if col != year_col:
                print(f"\n{col}:")
                print(f"  Range: [{df[col].min():.2f}, {df[col].max():.2f}]")
                print(f"  Mean: {df[col].mean():.2f}")
                print(f"  Std: {df[col].std():.2f}")
                print(f"  Missing: {df[col].isnull().sum()} ({df[col].isnull().sum()/len(df)*100:.1f}%)")
        print(f"{'='*60}\n")
        
        print(f"\nProcessing {len([c for c in df.columns if c != year_col])} data columns...\n")
    
    final_df = df.copy()

    # ===== Initialize statistics collection =====
    processing_stats = {
        'total_rows': len(df),
        'total_columns': len([c for c in df.columns if c != year_col]),
        'year_range': f"{int(df[year_col].min())} - {int(df[year_col].max())}",
        'total_outliers': 0,
        'columns_processed': 0,
        'column_stats': [],
        'missing_stats': missing_stats,
    }
    
    # Collect detection parameters
    if args.outlier_detection == 'trend_z':
        processing_stats['detection_params'] = f"z_threshold={args.z_threshold}"
    elif args.outlier_detection == 'isolation':
        processing_stats['detection_params'] = f"contamination={args.contamination}"
    elif args.outlier_detection == 'iqr':
        processing_stats['detection_params'] = f"iqr_multiplier={args.iqr_multiplier}"
    elif args.outlier_detection == 'modified_z':
        processing_stats['detection_params'] = f"mad_threshold={args.mad_threshold}"
    elif args.outlier_detection == 'zscore':
        processing_stats['detection_params'] = f"z_threshold={args.z_threshold}"
    
    # Collect smoothing parameters
    if args.smooth_method == 'ewma':
        processing_stats['smoothing_params'] = f"alpha={args.alpha}"
    elif args.smooth_method == 'rolling':
        processing_stats['smoothing_params'] = f"window={args.window}"
    elif args.smooth_method == 'savgol':
        processing_stats['smoothing_params'] = f"window={args.window}, polyorder={args.polyorder}"
    elif args.smooth_method == 'gaussian':
        processing_stats['smoothing_params'] = f"sigma={args.sigma}"
    else:
        processing_stats['smoothing_params'] = "-"
    
    # Prepare parameters for detection methods
    detection_kwargs = {
        'z_thresh': args.z_threshold,
        'contamination': args.contamination,
        'iqr_multiplier': args.iqr_multiplier,
        'mad_thresh': args.mad_threshold
    }
    
    # Prepare parameters for smoothing methods
    smoothing_kwargs = {
        'alpha': args.alpha,
        'window': args.window,
        'polyorder': args.polyorder,
        'sigma': args.sigma
    }
    
    # Process each column
    for col in df.columns:
        if col == year_col:
            continue
        
        if args.verbose:
            print(f">>> Processing column: {col}")
        # Store original statistics
        col_stat = {
            'name': col,
            'min_original': df[col].min(),
            'max_original': df[col].max(),
            'mean_original': df[col].mean(),
            'std_original': df[col].std(),
            'missing': df[col].isnull().sum(),
        }
        
        # 1. Outlier detection
        df_detected, method_note = detect_outliers(
            df.copy(), 
            col, 
            year_col, 
            method=args.outlier_detection,
            **detection_kwargs
        )
        
        outlier_count = df_detected["Outlier"].sum()

        col_stat['outliers'] = outlier_count
        col_stat['outlier_rate'] = (outlier_count / len(df_detected) * 100)
        processing_stats['total_outliers'] += outlier_count

        if args.verbose:
            outlier_pct = outlier_count / len(df_detected) * 100
            print(f"  Detected {outlier_count} outliers ({outlier_pct:.1f}%)")
        
        # 2. Outlier fixing
        df_fixed = fix_outliers(df_detected.copy(), col, method=args.fix_method)
        
        if args.verbose and outlier_count > 0:
            print(f"  Fixed outliers using '{args.fix_method}' method")
        
        # 3. Data smoothing
        df_smoothed = df_fixed.copy()
        if args.smooth_method != 'none':
            df_smoothed[col] = smooth_series(
                df_fixed[col], 
                method=args.smooth_method,
                **smoothing_kwargs
            )
            
            if args.verbose:
                print(f"  Applied '{args.smooth_method}' smoothing")
        
        # Store cleaned statistics
        col_stat['min_cleaned'] = df_smoothed[col].min()
        col_stat['max_cleaned'] = df_smoothed[col].max()
        col_stat['mean_cleaned'] = df_smoothed[col].mean()
        col_stat['std_cleaned'] = df_smoothed[col].std()
        
        processing_stats['column_stats'].append(col_stat)
        processing_stats['columns_processed'] += 1
        
        # 4. Visualization
        if not args.no_plot:
            plot_all(
                df, df_detected, df_fixed, df_smoothed,
                col, year_col, 
                method_note, 
                args.fix_method.capitalize(), 
                file_path, 
                args.smooth_method.upper()
            )
        
        final_df[col] = df_smoothed[col]
        
        if args.verbose:
            print(f"  [OK] Done\n")
    
    # Calculate overall statistics
    processing_stats['avg_outliers'] = processing_stats['total_outliers'] / processing_stats['columns_processed'] if processing_stats['columns_processed'] > 0 else 0
    processing_stats['outlier_rate'] = (processing_stats['total_outliers'] / (processing_stats['total_rows'] * processing_stats['columns_processed']) * 100) if processing_stats['columns_processed'] > 0 else 0
    
    # Save results
    output_paths = get_output_paths()
    DEFAULT_OUTPUT_DIR = output_paths['clean_data']
    output_dir = args.output if args.output else DEFAULT_OUTPUT_DIR
    
    # Create directory if it doesn't exist
    try:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            if args.verbose:
                print(f"[OK] Created output directory: {output_dir}")
    except Exception as e:
        print(f"[ERROR] Cannot create directory '{output_dir}': {e}")
        output_dir = os.path.dirname(file_path)
        print(f"[INFO] Using fallback directory: {output_dir}")
   
    # ===== Generate output filename based on input filename =====
    # Extract original filename without extension
    original_filename = os.path.basename(file_path)  # e.g., "Au_Barley_ha.xls"
    filename_without_ext = os.path.splitext(original_filename)[0]  # e.g., "Au_Barley_ha"
    
    # Create new filename with _cleaned suffix
    output_filename = f"{filename_without_ext}_cleaned.csv"
    
    cleaned_path = os.path.join(output_dir, output_filename)
    final_df.to_csv(cleaned_path, index=False)
    
    # Store output info in stats
    processing_stats['output_filename'] = output_filename
    processing_stats['figure_dir'] = os.path.join(output_paths['figures'], filename_without_ext)
    
    # Generate Markdown report
    md_report_path = generate_markdown_report(file_path, args, processing_stats, output_dir)
    print(f"[OK] Markdown report saved: {md_report_path}")
    
    # Generate TXT report
    txt_report_path = generate_txt_report(file_path, args, processing_stats, output_dir)
    print(f"[OK] TXT report saved: {txt_report_path}")
    
    print(f"\n{'='*60}")
    print(f"[SUCCESS] Processing completed!")
    print(f"{'='*60}")
    print(f"Summary:")
    print(f"  Columns processed: {processing_stats['columns_processed']}")
    print(f"  Total outliers detected: {processing_stats['total_outliers']}")
    print(f"  Overall outlier rate: {processing_stats['outlier_rate']:.2f}%")
    print(f"\nOutput files:")
    print(f"  Cleaned data: {cleaned_path}")
    print(f"  Markdown report: {md_report_path}")
    print(f"  TXT report: {txt_report_path}")
    print(f"{'='*60}\n")
    
    return final_df


# ========================================
# ===== Command Line Argument Parsing =====
# ========================================

def parse_arguments():
    """Parse command line arguments"""
    methods = get_available_methods()
    
    parser = argparse.ArgumentParser(
        description='DataProcess - Automated Data Cleaning and Outlier Detection Tool',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Usage Examples:
  python {os.path.basename(__file__)} -f data.xlsx
  python {os.path.basename(__file__)} -f data.xlsx -o trend_z -F median -s ewma
  python {os.path.basename(__file__)} -f data.xlsx -o isolation -F interpolate -s savgol -v
  python {os.path.basename(__file__)} --list-methods

Available Methods:
  Outlier Detection (-o): {', '.join(methods['detection'])}
  Outlier Fixing    (-F): {', '.join(methods['fixing'])}
  Data Smoothing    (-s): {', '.join(methods['smoothing'])}
        """
    )
    
    # Required arguments
    parser.add_argument('-f', '--file',
                        required=False,
                        default=None,
                        help='Input file path (.csv or .xlsx)')
    
    # ===== Outlier Detection Options =====
    detection_group = parser.add_argument_group('Outlier Detection Options')
    detection_group.add_argument('-o', '--outlier-detection',
                                 type=str,
                                 default='trend_z',
                                 metavar='METHOD',
                                 help=f"Outlier detection method (default: trend_z), options: {', '.join(methods['detection'])}")
    detection_group.add_argument('--z-threshold',
                                 type=float,
                                 default=2.8,
                                 metavar='FLOAT',
                                 help='Z-score threshold for Trend-Z and Z-score methods (default: 2.8)')
    detection_group.add_argument('--contamination',
                                 type=float,
                                 default=0.1,
                                 metavar='FLOAT',
                                 help='Contamination rate for IsolationForest (default: 0.1)')
    detection_group.add_argument('--iqr-multiplier',
                                 type=float,
                                 default=1.5,
                                 metavar='FLOAT',
                                 help='IQR multiplier (default: 1.5)')
    detection_group.add_argument('--mad-threshold',
                                 type=float,
                                 default=3.5,
                                 metavar='FLOAT',
                                 help='MAD threshold for Modified-Z method (default: 3.5)')
    
    # ===== Outlier Fixing Options =====
    fix_group = parser.add_argument_group('Outlier Fixing Options')
    fix_group.add_argument('-F', '--fix-method',
                          type=str,
                          default='median',
                          metavar='METHOD',
                          help=f"Outlier fixing method (default: median), options: {', '.join(methods['fixing'])}")
    
    # ===== Data Smoothing Options =====
    smooth_group = parser.add_argument_group('Data Smoothing Options')
    smooth_group.add_argument('-s', '--smooth-method',
                             type=str,
                             default='ewma',
                             metavar='METHOD',
                             help=f"Data smoothing method (default: ewma), options: {', '.join(methods['smoothing'])}")
    smooth_group.add_argument('--alpha',
                             type=float,
                             default=0.3,
                             metavar='FLOAT',
                             help='EWMA smoothing coefficient (default: 0.3)')
    smooth_group.add_argument('--window',
                             type=int,
                             default=3,
                             metavar='INT',
                             help='Window size for Rolling/Savgol (default: 3)')
    smooth_group.add_argument('--polyorder',
                             type=int,
                             default=2,
                             metavar='INT',
                             help='Polynomial order for Savgol filter (default: 2)')
    smooth_group.add_argument('--sigma',
                             type=float,
                             default=1.0,
                             metavar='FLOAT',
                             help='Standard deviation for Gaussian filter (default: 1.0)')
    
    # ===== Output Control =====
    output_group = parser.add_argument_group('Output Control')
    output_group.add_argument('-O', '--output',
                             type=str,
                             metavar='DIR',
                             help='Output directory (default: same as input file)')
    output_group.add_argument('--no-plot',
                             action='store_true',
                             help='Do not generate visualization plots')
    output_group.add_argument('-v', '--verbose',
                             action='store_true',
                             help='Show detailed processing information')
    
    # ===== Other Options =====
    parser.add_argument('--list-methods',
                       action='store_true',
                       help='List all available methods and exit')
    
    return parser.parse_args()


# ========================================
# ===== Main Entry Point =====
# ========================================

def main():
    """Main function"""
    args = parse_arguments()
    
    # List methods
    if args.list_methods:
        methods = get_available_methods()
        print("\n" + "="*60)
        print("Available Processing Methods")
        print("="*60)
        
        print("\n[Outlier Detection Methods] (-o, --outlier-detection)")
        for method in methods['detection']:
            func = DETECTION_METHODS[method]
            doc = func.__doc__.strip().split('\n')[0] if func.__doc__ else "No description"
            print(f"  * {method:15s} - {doc}")
        
        print("\n[Outlier Fixing Methods] (-F, --fix-method)")
        for method in methods['fixing']:
            func = FIXING_METHODS[method]
            doc = func.__doc__.strip() if func.__doc__ else "No description"
            print(f"  * {method:15s} - {doc}")
        
        print("\n[Data Smoothing Methods] (-s, --smooth-method)")
        for method in methods['smoothing']:
            func = SMOOTHING_METHODS[method]
            doc = func.__doc__.strip().split('\n')[0] if func.__doc__ else "No description"
            print(f"  * {method:15s} - {doc}")
        
        print("\n" + "="*60 + "\n")
        return 0
    
    # Check file existence
    if args.file is None:
        print(f"[ERROR] Argument -f/--file is required")
        return 1
    if not os.path.exists(args.file):
        print(f"[ERROR] File does not exist: '{args.file}'")
        return 1
    
    # Validate method names
    methods = get_available_methods()
    if args.outlier_detection not in methods['detection']:
        print(f"[ERROR] Unknown outlier detection method: '{args.outlier_detection}'")
        print(f"Available methods: {', '.join(methods['detection'])}")
        return 1
    
    if args.fix_method not in methods['fixing']:
        print(f"[ERROR] Unknown fixing method: '{args.fix_method}'")
        print(f"Available methods: {', '.join(methods['fixing'])}")
        return 1
    
    if args.smooth_method not in methods['smoothing']:
        print(f"[ERROR] Unknown smoothing method: '{args.smooth_method}'")
        print(f"Available methods: {', '.join(methods['smoothing'])}")
        return 1
    
    # Execute processing
    try:
        auto6_pipeline(args.file, args)
        return 0
    except Exception as e:
        print(f"\n[ERROR] Processing failed: {str(e)}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())