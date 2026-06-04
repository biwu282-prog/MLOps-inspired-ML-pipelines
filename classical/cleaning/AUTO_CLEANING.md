============================================================
Auto Data Cleaning Tool (Auto_Clean)
============================================================

[Project Overview]
This tool (Auto_Clean.py) is a "zero-configuration", fully automated data preprocessing script designed for time series and structured tabular data. It features a built-in decision engine that intelligently selects and executes the most appropriate outlier detection, outlier fixing, and data smoothing algorithms based on the statistical distribution characteristics of the data—requiring no manual hyperparameter tuning.

[Core Features]
* Auto Year Column Detection: Identifies the year column via regex matching or intelligent data range inference.
* Auto Missing Value Handling: Repairs missing data using Exponentially Weighted Moving Average (EWMA) combined with forward/backward fill.
* Auto Distribution Analysis: Identifies the shape of the data series (trend, normal, high-variance, or random).
* Adaptive Outlier Detection: Matches the algorithm to the data shape (e.g., Trend-Z, Standard Z-score, Isolation Forest, Modified MAD Z-score).
* Adaptive Outlier Fixing: Selects the optimal fixing strategy (median replacement or linear interpolation) based on the outlier ratio.
* Adaptive Data Smoothing: Evaluates data noise levels to trigger the appropriate smoothing filter (EWMA, Savitzky-Golay, or Rolling Average).
* Multi-Format I/O: Supports reading .csv, .xlsx, and .xls files, outputting a standardized .csv file for seamless downstream analysis.

[Dependencies]
The following Python libraries are required:
* pandas
* numpy
* scikit-learn (sklearn)
* scipy
* openpyxl (for modern Excel files)
* xlrd (for legacy Excel files)

[Command Line Usage]
Basic Syntax:
  python Auto_Clean.py -f <input_file_path> [options]

Arguments:
  -f, --file    [Required] Path to the input data file (.csv, .xlsx, .xls).
  -o, --output  [Optional] Output directory. If not specified, defaults to Result/Clean_data/ in the same directory as the script.
  -q, --quiet   [Optional] Quiet mode. Suppresses detailed processing logs in the console.

Examples:
  python Auto_Clean.py -f data.csv
  python Auto_Clean.py -f data.xlsx -o D:\Results\Clean_data
  python Auto_Clean.py -f data.csv -q

[Output Files]
Upon completion, two core files will be generated in the output directory:
1. Cleaned Dataset: <original_filename>_cleaned.csv, ready for feature engineering or model training.
2. Processing Report: <original_filename>_report.txt, detailing the distribution type, noise level, outlier rate, and the specific algorithms applied to each column.

============================================================