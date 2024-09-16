import streamlit as st
import joblib
import numpy as np
import re

def process(message):
    # Replace non-letter characters with space
    message = re.sub("[^a-zA-Z]", " ", message)
    # Convert to lowercase
    message = message.lower()
    return message

def vectorize(message):
    vectorizer = joblib.load('vectorizer.pkl')
    X = vectorizer.transform([message])  # Wrap the message in a list
    return X

def predict(X):
    model = joblib.load('multi_output_random_forest_model.pkl')
    y_pred = model.predict(X)
    return y_pred

def get_predicted_target(y_pred, target_variables):
    y_pred = y_pred.flatten()
    indices = np.where(y_pred == 1)[0]
    predicted_targets = [target_variables[i] for i in indices]
    return predicted_targets

target_variables = ['request', 'aid_related', 'medical_help', 'medical_products',
                    'search_and_rescue', 'security', 'military', 'water', 'food', 'shelter',
                    'clothing', 'money', 'missing_people', 'refugees', 'death', 'other_aid',
                    'infrastructure_related', 'transport', 'buildings', 'electricity',
                    'tools', 'hospitals', 'shops', 'aid_centers', 'other_infrastructure',
                    'weather_related', 'floods', 'storm', 'fire', 'earthquake', 'cold',
                    'other_weather']

def main():
    st.title("Disaster Response Message Classifier")
    st.write("Enter a message and the model will predict the category it belongs to.")

    # Text input
    message = st.text_area("Enter your message here", "")

    if st.button("Classify"):
        if message:
            # Preprocess the message
            processed_message = process(message)
            # Vectorize the message
            X = vectorize(processed_message)
            # Make prediction
            y_pred = predict(X)
            # Get predicted target(s)
            predicted_targets = get_predicted_target(y_pred, target_variables)
            if predicted_targets:
                st.success("Predicted Category(s):")
                for category in predicted_targets:
                    st.write(f"- {category}")
            else:
                st.warning("No category predicted.")
        else:
            st.warning("Please enter a message.")

if __name__ == '__main__':
    main()
