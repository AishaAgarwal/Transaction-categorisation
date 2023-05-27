import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_csv('Dataset.csv', delimiter = "\t")
print(df)

# Preprocess the data
X = pd.get_dummies(df[['Category', 'Transaction_Amount', 'Transaction_Date']])
y = df['Transaction_Type']

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Select a machine learning algorithm (Random Forest in this example)
model = RandomForestClassifier()

# Train the model
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
print(classification_report(y_test, y_pred))

# Visualize the results
# Bar chart - total amount comparison
df['Transaction_Type'] = y
df.groupby('Transaction_Type')['Transaction_Amount'].sum().plot(kind='bar')
plt.xlabel('Transaction Type')
plt.ylabel('Total Amount')
plt.title('Expenditure vs. Income')
plt.show()

# Count of transactions by category
df['Predicted_Type'] = model.predict(X)
df_expenditure = df[df['Predicted_Type'] == 'expenditure']
df_expenditure['Category'].value_counts().plot(kind='bar')
plt.xlabel('Category')
plt.ylabel('Number of Transactions')
plt.title('Number of Expenditure Transactions by Category')
plt.show()