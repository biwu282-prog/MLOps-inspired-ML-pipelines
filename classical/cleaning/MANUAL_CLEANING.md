============================================================
Custom Data Processing Workstation (Manual_Clean)
============================================================

[Project Overview]
This tool (Manual_Clean.py) is a highly modular, parameter-driven "white-box" data preprocessing framework. Built upon an elegant method registration system, it encapsulates various outlier detection, fixing, and smoothing algorithms into independent modules. Users have granular control over every stage of the data cleaning pipeline via command-line arguments, including the fine-tuning of algorithm-specific hyperparameters. It is ideal for rigorous feature engineering preparation where transparent data quality control is paramount.

[Core Features]
* Pluggable Architecture: Built-in library of classic algorithms, allowing free combination of detection, fixing, and smoothing pipelines via CLI.
* Pre-emptive Missing Value Handling: Automatically repairs global missing values using EWMA (Exponentially Weighted Moving Average) prior to core processing, ensuring time-series continuity.
* Granular Hyperparameter Tuning: Supports custom adjustments for Z-score thresholds, Isolation Forest contamination rates, smoothing window sizes, and more.
* Visual Diagnostics: Automatically generates multi-stage visual comparisons (original vs. detected outliers vs. smoothed) for each feature (can be disabled for performance).
* In-depth Quality Reporting: Outputs comprehensive Markdown and TXT diagnostic reports per run, tracking mean shifts, outlier rates, and algorithm execution logs.

[Dependencies]
The following Python libraries are required:
* pandas
* numpy
* scikit-learn (sklearn)
* scipy
* matplotlib (for generating diagnostic plots)
* openpyxl / xlrd (for parsing Excel/CSV files)

[Command Line Usage]
Basic Syntax:
  python Manual_Clean.py -f <input_file_path> [algorithm_options] [hyperparameters] [output_controls]

Utility Commands:
  python Manual_Clean.py --list-methods    # List all registered algorithm modules and descriptions
  python Manual_Clean.py -h                # View detailed help and hyperparameter descriptions

Examples:
1. Basic Run (uses default algorithm pipeline):
   python Manual_Clean.py -f data.csv

2. Custom Pipeline (Isolation Forest + Linear Interpolation + Savitzky-Golay Smoothing):
   python Manual_Clean.py -f data.xlsx -o isolation -F interpolate -s savgol

3. Advanced Hyperparameter Tuning (Strict Z-score + larger rolling window) with verbose logs:
   python Manual_Clean.py -f data.csv -o zscore --z-threshold 3.5 -s rolling --window 5 -v

4. Fast Mode (Skip plot generation, output data and reports only):
   python Manual_Clean.py -f data.xlsx --no-plot

[Output Files]
Upon completion, a Result directory will be generated in the same directory as the input file (or the path specified by `-O`), containing:
1. /Clean_data/xxx_cleaned.csv: The final processed dataset in standardized CSV format.
2. /Clean_data/xxx_report.md & .txt: Detailed preprocessing diagnostic reports evaluating statistical variance.
3. /Figure/xxx/: A directory containing .png multi-stage visual comparison charts for every processed feature column.

============================================================