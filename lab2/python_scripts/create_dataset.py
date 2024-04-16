import pandas as pd
import os
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split

# Data loading
wine = load_wine()
X = wine.data  # type: ignore
y = wine.target  # type: ignore

# Transforming data in DataFrame
df = pd.DataFrame(data=X, columns=wine.feature_names)  # type: ignore
df['target'] = y

# Splitting the DataFrame into training and test samples
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

# Create directories for data storage
os.makedirs('train', exist_ok=True)
os.makedirs('test', exist_ok=True)

# Saving training and test samples to CSV files
train_df.to_csv('train/wine_train.csv', index=False)
test_df.to_csv('test/wine_test.csv', index=False)
