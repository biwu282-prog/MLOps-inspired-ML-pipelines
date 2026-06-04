# Classical Dataset

`barley_data.csv` is the small annual structured dataset used by the classical
machine learning pipeline.

The columns are:

- `Year`
- `output_ha`
- `yield_kg_ha`
- `ave_temp`
- `Precipitation_mm`

The agricultural and climate variables were prepared from FAOSTAT and NASA
POWER data sources.

The final manual-cleaning configuration used by the pipeline is:

```text
Outlier detection: z-score, threshold 3.5
Outlier fixing: median replacement
Smoothing: rolling mean, window 5
```
