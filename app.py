from flask import Flask, render_template, request, jsonify
import joblib
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Initialize Flask application
app = Flask(__name__)

# Load the trained model and scaler
best_rf_model = joblib.load('models/random_forest_model.pkl')
scaler = joblib.load('models/scaler.pkl')

# List of input features for the prediction
input_columns = [
    'Application mode', 'Application order', 'Course', 'Previous qualification (grade)', 
    'Mother\'s qualification', 'Father\'s qualification', 'Mother\'s occupation', 
    'Father\'s occupation', 'Admission grade', 'Debtor', 'Tuition fees up to date', 
    'Scholarship holder', 'Age at enrollment', 'Curricular units 1st sem (enrolled)', 
    'Curricular units 1st sem (evaluations)', 'Curricular units 1st sem (approved)', 
    'Curricular units 1st sem (grade)', 'Curricular units 2nd sem (enrolled)', 
    'Curricular units 2nd sem (evaluations)', 'Curricular units 2nd sem (approved)', 
    'Curricular units 2nd sem (grade)', 'Unemployment rate', 'Inflation rate', 'GDP'
]

# Define the prediction route
@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        # Retrieve input values from the form
        input_data = []
        for column in input_columns:
            input_value = request.form.get(column)
            input_data.append(input_value)

        # Convert input data to a DataFrame
        input_df = pd.DataFrame([input_data], columns=input_columns)

        # Step 1: Encode categorical columns (same as the test data)
        categorical_columns = ['Application mode', 'Course', 'Previous qualification (grade)', 
                               'Mother\'s qualification', 'Father\'s qualification', 
                               'Mother\'s occupation', 'Father\'s occupation', 
                               'Debtor', 'Tuition fees up to date', 'Scholarship holder']
        
        # Apply LabelEncoder
        label_encoders = {}
        for col in categorical_columns:
            le = LabelEncoder()
            input_df[col] = le.fit_transform(input_df[col].astype(str))  # Encoding as string
            label_encoders[col] = le  # Save the label encoder for future use

        # Step 2: Scale numerical data
        input_scaled = scaler.transform(input_df)

        # Step 3: Predict using the trained model
        prediction = best_rf_model.predict(input_scaled)

        # Step 4: Map predictions (assuming 0 = dropout, 1 = graduate)
        prediction_labels = ['Dropout', 'Graduate']
        result = prediction_labels[prediction[0]]

        return render_template('index.html', result=result)

    return render_template('index.html')

# Run the app
if __name__ == '__main__':
    app.run(debug=True)
