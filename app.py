from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
import nltk
import re
from nltk.sentiment import SentimentIntensityAnalyzer
from collections import defaultdict

# Initialize Flask app
app = Flask(__name__)

# Download necessary NLTK data
nltk.download('vader_lexicon')

# Load Dataset
try:
    df = pd.read_csv("data.csv", low_memory=False, dtype={4: str, 5: str})
except FileNotFoundError:
    print("Error: Dataset file not found.")
    exit()

# Drop missing reviews
df.dropna(subset=['Reviews'], inplace=True)

# Initialize Sentiment Analyzer
sia = SentimentIntensityAnalyzer()

# Vectorized Sentiment Analysis
df['Sentiment_Score'] = df['Reviews'].astype(str).apply(lambda review: sia.polarity_scores(review)['compound'])
df['Sentiment'] = np.where(df['Sentiment_Score'] > 0, "Positive",
                           np.where(df['Sentiment_Score'] < 0, "Negative", "Neutral"))

# Efficient mapping
sentiment_mapping = {"Negative": 0, "Neutral": 1, "Positive": 2}
df['Sentiment_Label'] = df['Sentiment'].map(sentiment_mapping)

# Extract unique conditions
unique_conditions = set(map(str.lower, df['Condition'].dropna().unique()))

# Symptom-to-disease mapping
symptom_disease_map = {
    "fever": ["flu", "common cold", "covid-19"],
    "cough": ["common cold", "flu", "bronchitis"],
    "headache": ["migraine", "flu", "tension headache"],
    "sore throat": ["strep throat", "common cold", "flu"],
    "sneezing": ["allergy", "common cold", "sinusitis"]
}

def predict_disease(symptoms):
    disease_counts = defaultdict(int)
    for symptom in symptoms:
        for disease in symptom_disease_map.get(symptom, []):
            disease_counts[disease] += 1

    if not disease_counts:
        return "Unclear diagnosis. You may have symptoms of multiple diseases. Consult a professional."

    sorted_diseases = sorted(disease_counts.items(), key=lambda x: x[1], reverse=True)
    top_diseases = sorted_diseases[:2]

    if len(top_diseases) > 1 and top_diseases[0][1] != top_diseases[1][1]:
        total_symptoms = sum(disease_counts.values())
        prediction = "You may have "
        prediction += ", ".join([f"{disease} with {count/total_symptoms*100:.1f}% probability" for disease, count in top_diseases])
        prediction += ". It is better to consult a doctor."
        return prediction

    return f"You may have {top_diseases[0][0]}. It's recommended to consult a doctor for confirmation."

def extract_condition(user_input):
    lower_input = user_input.lower()
    return next((condition for condition in unique_conditions if condition in lower_input), None)

def recommend_medicine(user_input):
    condition = extract_condition(user_input)
    symptoms = re.findall(r'\b\w+\b', user_input.lower())
    
    disease_prediction = predict_disease(symptoms)
    
    if not condition:
        message = "No specific condition matched based on your symptoms, but here's a disease prediction: " + disease_prediction
        return {
            "Disease Prediction": disease_prediction,
            "Message": message
        }

    condition_drugs = df[df['Condition'].str.contains(condition, case=False, na=False)]
    
    if condition_drugs.empty:
        return {
            "Disease Prediction": disease_prediction,
            "Message": "No medicine found for this condition. " 
        }

    best_drug = condition_drugs.groupby("Drug")["Sentiment_Label"].mean().idxmax()

    message = f"Use {best_drug} to cure your disease. Hope you recover soon!"
    
    effectiveness = condition_drugs.loc[condition_drugs['Drug'] == best_drug, 'Effectiveness'].mean()
    effectiveness = f"{effectiveness:.1f}%" if not np.isnan(effectiveness) else "Unknown"
    
    common_side_effects = condition_drugs.loc[condition_drugs['Drug'] == best_drug, 'Sides'].mode()
    common_side_effects = common_side_effects.iloc[0] if not common_side_effects.empty else "Not available"
    
    user_review_sentiment = "Mostly Positive" if effectiveness != "Unknown" and float(effectiveness.strip('%')) > 70 else "Mixed"
    
    return {
        "Best Drug": best_drug,
        "Effectiveness": effectiveness,
        "Side Effects": common_side_effects,
        "User Reviews Sentiment": user_review_sentiment,
        "Disease Prediction": disease_prediction,
        "Message": message
    }

@app.route('/')
def index():
    return open('index.html', 'r').read()

@app.route('/recommend', methods=['POST'])
def recommend():
    data = request.get_json()
    symptoms = data['symptoms']
    recommendation = recommend_medicine(symptoms)
    return jsonify(recommendation)

if __name__ == '__main__':
    app.run(debug=True)
