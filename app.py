from flask import Flask, request, jsonify
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

app = Flask(__name__)

# Endpoint for training and making predictions
@app.route('/predict', methods=['POST'])
def predict():
    # Get the data from the request
    data = request.json
    
    # Load the dataset
    df = pd.DataFrame(data)
    
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
    report = classification_report(y_test, y_pred)
    
    # Return the classification report as a JSON response
    return jsonify({'classification_report': report})

if __name__ == '__main__':
    app.run()
