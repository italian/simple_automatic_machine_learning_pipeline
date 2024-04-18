import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)
import pickle

# Loading test data
X_test = pd.read_csv('data/test/X_test.csv')
y_test = pd.read_csv('data/test/y_test.csv')

# Loading a model from a file
with open('models/model.pkl', 'rb') as file:
    model = pickle.load(file)

# Prediction on test data
y_pred = model.predict(X_test)

# Calculating model quality metrics
accuracy = accuracy_score(y_test['target'], y_pred)
precision = precision_score(y_test['target'], y_pred, average='weighted')
recall = recall_score(y_test['target'], y_pred, average='weighted')
f1 = f1_score(y_test['target'], y_pred, average='weighted')

# Creating a DataFrame for the results
results = pd.DataFrame({
    'Accuracy': [accuracy],
    'Precision': [precision],
    'Recall': [recall],
    'F1-score': [f1]
})

print('Results on test data:')
print(results.to_string(index=False))
