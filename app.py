import streamlit as st
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import pandas as pd
import pickle

# Load the trained model
model = tf.keras.models.load_model('model.h5', compile=False)

# Load the encoders and scaler
with open('label_encoder_gender.pkl', 'rb') as file:
    label_encoder_gender = pickle.load(file)

with open('onehot_encoder_geo.pkl', 'rb') as file:
    onehot_encoder_geo = pickle.load(file)

with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)


## streamlit app
st.title('Customer Churn PRediction')

# User input
geography = st.selectbox('Geography', onehot_encoder_geo.categories_[0])#default value is first value in the list
gender = st.selectbox('Gender', label_encoder_gender.classes_)#from the possible values of gender
age = st.slider('Age', 18, 92)
balance = st.number_input('Balance')
credit_score = st.number_input('Credit Score')
estimated_salary = st.number_input('Estimated Salary')
tenure = st.slider('Tenure', 0, 10)
num_of_products = st.slider('Number of Products', 1, 4)
has_cr_card = st.selectbox('Has Credit Card', [0, 1])
is_active_member = st.selectbox('Is Active Member', [0, 1])

# Prepare the input data in the form of dataframe.
input_data = pd.DataFrame({
    'CreditScore': [credit_score],
    'Gender': [label_encoder_gender.transform([gender])[0]],#label encoding the gender and it expects a 1d array like during training.
    'Age': [age],
    'Tenure': [tenure],
    'Balance': [balance],
    'NumOfProducts': [num_of_products],
    'HasCrCard': [has_cr_card],
    'IsActiveMember': [is_active_member],
    'EstimatedSalary': [estimated_salary]
})

# One-hot encode 'Geography' -> again give 2d array [[geography]].
geo_encoded = onehot_encoder_geo.transform([[geography]]).toarray()
geo_encoded_df = pd.DataFrame(geo_encoded, columns=onehot_encoder_geo.get_feature_names_out(['Geography']))

# Combine one-hot encoded columns with input data
#suppose your original input_data looked like this with index values like 5, 8, 10 instead of 0,1,2 (which is very common after filtering or splitting), containing columns such as Age, Salary, and Geography; then after applying One-Hot Encoding, you create geo_encoded_df which contains columns like Geo_France, Geo_Germany, Geo_Spain but with a fresh index starting from 0,1,2; now if you directly concatenate using pd.concat([input_data, geo_encoded_df], axis=1), the result becomes completely wrong because pandas aligns rows based on index, so indices 5,8,10 do not match with 0,1,2, leading to NaN values and corrupted data where rows are misaligned; to fix this, you apply input_data.reset_index(drop=True) which resets the index of input_data to 0,1,2 so that it matches the index of geo_encoded_df, and now when you concatenate using pd.concat([input_data.reset_index(drop=True), geo_encoded_df], axis=1), the result is perfectly aligned with correct rows and no NaN values; additionally, the reason we specifically use drop=True is that if we only used reset_index() without drop, the old index (5,8,10) would be added as a new column called index, which is useless, can confuse the ML model, and pollutes the dataset with irrelevant features; therefore, reset_index(drop=True) ensures proper row alignment and avoids adding an unnecessary column, making concatenation correct and clean.
input_data = pd.concat([input_data.reset_index(drop=True), geo_encoded_df], axis=1)

# Scale the input data
input_data_scaled = scaler.transform(input_data)


# Predict churn
prediction = model.predict(input_data_scaled)
prediction_proba = prediction[0][0]

st.write(f'Churn Probability: {prediction_proba:.2f}')

if prediction_proba > 0.5:
    st.write('The customer is likely to churn.')
else:
    st.write('The customer is not likely to churn.')
