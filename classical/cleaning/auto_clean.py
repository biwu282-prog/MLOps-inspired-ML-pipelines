import pandas as pd
import numpy as np
import re
import os
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import IsolationForest
from scipy.stats import zscore, shapiro, jarque_bera
from scipy.signal import savgol_filter
import matplotlib.pyplot as plt

# ========================================
# ===== Data Analysis & Decision Engine =====
# ========================================

def analyze_data_distribution(series):
    """
    Analyze data distribution type
    
    Returns:
    --------
    str : Distribution type ('normal', 'trend', 'seasonal', 'random')
    """
    # Remove NaN
    clean_series = series.dropna()
    
    if len(clean_series) < 8:
        return 'random'
    
    # Test for trend
    x = np.arange(len(clean_series))
    y = clean_series.values
    correlation = np.corrcoef(x, y)[0, 1]
    
    if abs(correlation) > 0.7:
        return 'trend'
    
    # Test for normality
    try:
        _, p_value = shapiro(clean_series)
        if p_value > 0.05:
            return 'normal'
    except:
        pass
    
    # Check coefficient of variation
    cv = clean_series.std() / clean_series.mean() if clean_series.mean() != 0 else 0
    
    if cv > 0.5:
        return 'high_variance'
    
    return 'random'


def calculate_noise_level(series):
    """
    Calculate noise level in data
    
    Returns:
    --------
    float : Noise level (0-1, higher means more noise)
    """
    if len(series) < 3:
        return 0
    
    # Calculate first differences
    diff = np.diff(series.dropna())
    
    if len(diff) == 0:
        return 0
    
    # Noise metric: ratio of diff std to original std
    noise = np.std(diff) / (np.std(series.dropna()) + 1e-10)
    
    return min(noise, 1.0)


def auto_select_detection_method(series, year_series):
    """
    Automatically select outlier detection method based on data characteristics
    
    Returns:
    --------
    str : Method name
    dict : Method parameters
    """
    dist_type = analyze_data_distribution(series)
    
    # Decision rules
    if dist_type == 'trend':
        return 'trend_z', {'z_thresh': 2.8}
    elif dist_type == 'normal':
        return 'zscore', {'z_thresh': 3.0}
    elif dist_type == 'high_variance':
        return 'isolation', {'contamination': 0.1}
    else:
        return 'modified_z', {'mad_thresh': 3.5}


def auto_select_fixing_method(outlier_count, outlier_rate):
    """
    Automatically select fixing method based on outlier characteristics
    
    Returns:
    --------
    str : Method name
    """
    if outlier_rate > 20:
        return 'median'
    return 'interpolate'


def auto_select_smoothing_method(series, noise_level, dist_type):
    """
    Automatically decide if smoothing is needed and select method
    
    Returns:
    --------
    str : Method name ('none' if no smoothing needed)
    dict : Method parameters
    """
    # Decide if smoothing is needed
    if noise_level < 0.15:
        return 'none', {}
    
    # Select smoothing method based on characteristics
    if dist_type == 'trend':
        return 'ewma', {'alpha': 0.3}
    elif noise_level > 0.4:
        return 'savgol', {'window': 5, 'polyorder': 2}
    else:
        return 'rolling', {'window': 3}


# ========================================
# ===== Processing Methods (Simplified) =====
# ========================================

def detect_outliers_auto(df, col, year_col, method, params):
    """Unified outlier detection"""
    if method == 'trend_z':
        X = df[year_col].values.reshape(-1, 1)
        y = df[col].values
        lr = LinearRegression().fit(X, y)
        trend_pred = lr.predict(X)
        resid = y - trend_pred
        z = (resid - resid.mean()) / (resid.std(ddof=0) + 1e-10)
        df["Outlier"] = (np.abs(z) > params['z_thresh']).astype(int)
        df[f"{col}_TrendPred"] = trend_pred
        
    elif method == 'zscore':
        z = np.abs(zscore(df[col]))
        df["Outlier"] = (z > params['z_thresh']).astype(int)
        
    elif method == 'isolation':
        iso = IsolationForest(contamination=params['contamination'], random_state=42)
        preds = iso.fit_predict(df[[col]].values.reshape(-1, 1))
        df["Outlier"] = (preds == -1).astype(int)
        
    elif method == 'modified_z':
        median = df[col].median()
        mad = np.median(np.abs(df[col] - median))
        if mad == 0:
            modz = np.zeros(len(df))
        else:
            modz = 0.6745 * (df[col] - median) / mad
        df["Outlier"] = (np.abs(modz) > params['mad_thresh']).astype(int)
    
    return df


def fix_outliers_auto(df, col, method):
    """Unified outlier fixing"""
    df[col] = df[col].astype(float)
    
    if method == 'median':
        normal_data = df.loc[df["Outlier"] == 0, col]
        if len(normal_data) > 0:
            median_val = normal_data.median()
            df.loc[df["Outlier"] == 1, col] = median_val
            
    elif method == 'interpolate':
        df.loc[df["Outlier"] == 1, col] = np.nan
        df[col] = df[col].interpolate(method='linear', limit_direction='both')
    
    return df


def smooth_data_auto(series, method, params):
    """Unified smoothing"""
    if method == 'none':
        return series
    elif method == 'ewma':
        return series.ewm(alpha=params['alpha'], adjust=False).mean()
    elif method == 'rolling':
        return series.rolling(window=params['window'], min_periods=1, center=True).mean()
    elif method == 'savgol':
        window = params['window']
        if window % 2 == 0:
            window += 1
        try:
            smoothed = savgol_filter(series, window, params['polyorder'])
            return pd.Series(smoothed, index=series.index)
        except:
            return series.rolling(window=3, min_periods=1, center=True).mean()
    
    return series


def handle_missing_ewma(series, alpha=0.3):
    """Handle missing values with EWMA"""
    missing_mask = series.isnull()
    if missing_mask.sum() == 0:
        return series
    
    filled = series.copy()
    filled = series.ffill().bfill()
    ewma_smoothed = filled.ewm(alpha=alpha, adjust=False).mean()
    
    result = series.copy()
    result[missing_mask] = ewma_smoothed[missing_mask]
    return result


# ========================================
# ===== Auto Processing Pipeline =====
# ========================================

def auto_process_data(file_path, output_dir=None, verbose=True):
    """
    Fully automatic data processing pipeline
    
    Parameters:
    -----------
    file_path : str
        Input Excel file path
    output_dir : str
        Output directory (optional)
    verbose : bool
        Show detailed processing information
    
    Returns:
    --------
    DataFrame : Cleaned dataframe
    dict : Processing report
    """
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"AUTO DATA PROCESSING")
        print(f"{'='*60}\n")
    
    # ===== 1. Load Data =====
    if verbose:
        print(f"[1/6] Loading data...")
    
    if file_path.endswith('.csv'):
        df = pd.read_csv(file_path)
    elif file_path.endswith('.xlsx'):
        df = pd.read_excel(file_path, engine='openpyxl')
    elif file_path.endswith('.xls'):
        df = pd.read_excel(file_path, engine='xlrd')
    else:
        raise ValueError(f"Unsupported file format: {file_path}. Please use .csv, .xlsx, or .xls")
    
    if verbose:
        print(f"  ✓ Loaded {df.shape[0]} rows × {df.shape[1]} columns\n")
    
    # ===== 2. Detect Year Column =====
    year_cols = [col for col in df.columns if re.search("year", col, re.IGNORECASE)]
    year_col = year_cols[0] if year_cols else None
    
    if not year_col:
        for col in df.columns:
            if np.issubdtype(df[col].dtype, np.number):
                if df[col].between(1900, 2100).sum() > len(df) * 0.6:
                    year_col = col
                    break
    
    if not year_col:
        raise ValueError("No year column detected")
    
    # ===== 3. Select Numeric Columns =====
    numeric_cols = [col for col in df.columns 
                    if np.issubdtype(df[col].dtype, np.number) 
                    and col != year_col 
                    and df[col].nunique() > 3]
    
    df = df[[year_col] + numeric_cols].copy()
    df = df.dropna(how="all").drop_duplicates()
    
    if verbose:
        print(f"[2/6] Data preparation")
        print(f"  ✓ Year column: {year_col}")
        print(f"  ✓ Data columns: {numeric_cols}\n")
    
    # ===== 4. Handle Missing Values =====
    if verbose:
        print(f"[3/6] Handling missing values...")
    
    total_missing = df[numeric_cols].isnull().sum().sum()
    missing_info = {}
    
    if total_missing > 0:
        for col in numeric_cols:
            if df[col].isnull().sum() > 0:
                missing_info[col] = df[col].isnull().sum()
                df[col] = handle_missing_ewma(df[col], alpha=0.3)
        
        if verbose:
            print(f"  ✓ Filled {total_missing} missing values using EWMA\n")
    else:
        if verbose:
            print(f"  ✓ No missing values found\n")
    
    # ===== 5. Process Each Column =====
    if verbose:
        print(f"[4/6] Analyzing and processing columns...\n")
    
    final_df = df.copy()
    processing_report = {
        'columns': {},
        'methods_used': {}
    }
    
    for col in numeric_cols:
        if verbose:
            print(f"  Processing: {col}")
        
        # Analyze data characteristics
        dist_type = analyze_data_distribution(df[col])
        noise_level = calculate_noise_level(df[col])
        
        if verbose:
            print(f"    - Distribution: {dist_type}")
            print(f"    - Noise level: {noise_level:.2f}")
        
        # Auto-select detection method
        detect_method, detect_params = auto_select_detection_method(df[col], df[year_col])
        
        if verbose:
            print(f"    - Detection method: {detect_method}")
        
        # Detect outliers
        df_detected = detect_outliers_auto(df.copy(), col, year_col, detect_method, detect_params)
        outlier_count = df_detected["Outlier"].sum()
        outlier_rate = (outlier_count / len(df_detected) * 100)
        
        if verbose:
            print(f"    - Outliers found: {outlier_count} ({outlier_rate:.1f}%)")
        
        # Auto-select fixing method
        fix_method = auto_select_fixing_method(outlier_count, outlier_rate)
        
        if verbose and outlier_count > 0:
            print(f"    - Fixing method: {fix_method}")
        
        # Fix outliers
        df_fixed = fix_outliers_auto(df_detected.copy(), col, fix_method)
        
        # Auto-select smoothing method
        smooth_method, smooth_params = auto_select_smoothing_method(df_fixed[col], noise_level, dist_type)
        
        if verbose:
            if smooth_method != 'none':
                print(f"    - Smoothing: {smooth_method}")
            else:
                print(f"    - Smoothing: not needed")
        
        # Apply smoothing
        final_df[col] = smooth_data_auto(df_fixed[col], smooth_method, smooth_params)
        
        # Record processing info
        processing_report['columns'][col] = {
            'distribution': dist_type,
            'noise_level': noise_level,
            'outliers': outlier_count,
            'outlier_rate': outlier_rate,
            'detection_method': detect_method,
            'fixing_method': fix_method if outlier_count > 0 else 'none',
            'smoothing_method': smooth_method
        }
        
        if verbose:
            print(f"    ✓ Done\n")
    
    # ===== 6. Save Results =====
    if verbose:
        print(f"[5/6] Saving results...")
    
    if output_dir is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        output_dir = os.path.join(script_dir, 'Result', 'Clean_data')
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Save cleaned data
    base_name = os.path.splitext(os.path.basename(file_path))[0]
    output_file = os.path.join(output_dir, f"{base_name}_auto_cleaned.csv")
    final_df.to_csv(output_file, index=False)
    
    # Save processing report
    report_file = os.path.join(output_dir, f"{base_name}_report.txt")
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("="*60 + "\n")
        f.write("AUTO PROCESSING REPORT\n")
        f.write("="*60 + "\n\n")
        
        f.write(f"Input File: {file_path}\n")
        f.write(f"Output File: {output_file}\n\n")
        
        f.write("Processing Summary:\n")
        f.write("-"*60 + "\n")
        
        for col, info in processing_report['columns'].items():
            f.write(f"\n{col}:\n")
            f.write(f"  Distribution Type: {info['distribution']}\n")
            f.write(f"  Noise Level: {info['noise_level']:.2f}\n")
            f.write(f"  Outliers Detected: {info['outliers']} ({info['outlier_rate']:.1f}%)\n")
            f.write(f"  Detection Method: {info['detection_method']}\n")
            f.write(f"  Fixing Method: {info['fixing_method']}\n")
            f.write(f"  Smoothing Method: {info['smoothing_method']}\n")
    
    if verbose:
        print(f"  ✓ Cleaned data: {output_file}")
        print(f"  ✓ Report: {report_file}\n")
        
        print(f"[6/6] Complete!\n")
        print(f"{'='*60}")
        print(f"PROCESSING COMPLETED")
        print(f"{'='*60}\n")
    
    return final_df, processing_report


# ========================================
# ===== Main Entry Point =====
# ========================================

def main():
    """Main function for command line usage"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Auto Data Processing - Fully Automatic Data Cleaning Tool',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python Auto_Clean.py -f data.csv/xlsx
  python Auto_Clean.py -f data.csv/xlsx -o D:\\Results
  python Auto_Clean.py -f data.csv/xlsx -q

Features:
  • Automatic missing value handling (EWMA)
  • Automatic distribution analysis
  • Automatic outlier detection method selection
  • Automatic outlier fixing
  • Automatic smoothing decision and method selection
        """
    )
    
    parser.add_argument('-f', '--file',
                       required=True,
                       help='Input csv/Excel file path')
    
    parser.add_argument('-o', '--output',
                       type=str,
                       help='Output directory (optional)')
    
    parser.add_argument('-q', '--quiet',
                       action='store_true',
                       help='Quiet mode (minimal output)')
    
    args = parser.parse_args()
    
    # Check file existence
    if not os.path.exists(args.file):
        print(f"[ERROR] File does not exist: '{args.file}'")
        return 1
    
    try:
        auto_process_data(args.file, output_dir=args.output, verbose=not args.quiet)
        return 0
    except Exception as e:
        print(f"\n[ERROR] Processing failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())