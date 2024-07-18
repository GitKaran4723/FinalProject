import pandas as pd

def preprocess_data(file):
    data = pd.read_csv(file)
    new_columns = data.columns.tolist()
    return data, new_columns

def detect_trends(data, columns):
    trend_data = data.copy()
    for column in columns:
        if pd.api.types.is_numeric_dtype(trend_data[column]):
            trend_data[column + '_trend'] = trend_data[column].rolling(window=5, min_periods=1).mean()
    return trend_data
