import pandas as pd
import os
from sklearn.datasets import load_wine

# Data loading
wine = load_wine()
X = wine.data  # type: ignore
y = wine.target  # type: ignore

# Transforming data in DataFrame
df = pd.DataFrame(data=X, columns=wine.feature_names)  # type: ignore
df['target'] = y

print(df.info())
print(df.describe())

# Create directories for data storage
os.makedirs('data', exist_ok=True)

# Saving data to CSV files
df.to_csv('data/wine.csv', index=False)
