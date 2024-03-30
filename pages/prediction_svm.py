import streamlit as st
import pandas as pd 
import pickle 
from modelling import SupportVectorMachine as svm  # Assuming SVM model is implemented in 'SupportVectorMachine.py'

st.title('Death Prediction using Support Vector Machine')

# Function to load the model and output data from pickle files
def load_model_and_output(model_file, output_file):
    with open(model_file, 'rb') as model_pickle, open(output_file, 'rb') as output_pickle:
        model = pickle.load(model_pickle)
        output = pickle.load(output_pickle)
    return model, output

# Load the trained SVM model and output data
model_file_path = 'svm_model.pickle'
output_file_path = 'svm_output.pickle'
svm_model, uniques = load_model_and_output(model_file_path, output_file_path)

# Load the dataset
data = pd.read_csv('cause_of_deaths.csv')

# Filter options
min_year = int(data['Year'].min())
max_year = int(data['Year'].max())
selected_year = st.slider('Select year range:', min_year, max_year, (2019, max_year))
selected_country = st.selectbox('Select country:', data['Country/Territory'].unique())
selected_disease = st.selectbox('Select disease:', data.columns[3:])  # Assuming disease columns start from the 4th column

# Filter data based on user selections
filtered_data = data[(data['Year'] >= selected_year[0]) & (data['Year'] <= selected_year[1]) &
                     (data['Country/Territory'] == selected_country) &
                     (data[selected_disease] == 1)]

if not filtered_data.empty:
    # Make predictions using the loaded model
    predictions = svm_model.predict(filtered_data)

    # Display the predictions
    st.write(predictions)
else:
    st.write('No data available for the selected filters.')
