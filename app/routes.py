from flask import render_template, request, jsonify
import pandas as pd
import joblib
from app import app
from preprocess import preprocess_data

# Load the trained model
model = joblib.load('model/credit_model.pkl')

def generate_recommendations(user_data, model):
    # Predict potential credit score improvement
    predicted_score = model.predict(user_data)

    # Generate recommendations
    recommendations = []
    if predicted_score < 600:
        recommendations.append("Consider paying down high-interest debt.")
        recommendations.append("Check for errors on your credit report.")
    elif predicted_score < 700:
        recommendations.append("Diversify your credit accounts.")
        recommendations.append("Keep credit card balances low.")
    else:
        recommendations.append("Maintain good credit habits.")
        recommendations.append("Avoid opening new credit accounts frequently.")

    return recommendations

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/recommend', methods=['POST'])
def recommend():
    user_data = request.json
    print("Received user data:", user_data)  # Debugging statement
    user_df = pd.DataFrame(user_data, index=[0])
    processed_user_data = preprocess_data(user_df)
    print("Processed user data:", processed_user_data)  # Debugging statement
    recommendations = generate_recommendations(processed_user_data, model)
    return jsonify(recommendations)