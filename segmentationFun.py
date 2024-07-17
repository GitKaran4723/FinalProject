import pandas as pd

def preprocess_data(file):
    data = pd.read_csv(file)
    missing_columns = [col for col in ['CustomerID', 'Quantity', 'UnitPrice'] if col not in data.columns]
    if missing_columns:
        return None, f"Missing required columns: {', '.join(missing_columns)}"
    data['TotalSpend'] = data['Quantity'] * data['UnitPrice']
    return data, None
