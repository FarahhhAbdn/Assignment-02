import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
import pickle 
from sklearn.model_selection import train_test_split 
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import LabelEncoder
import streamlit as st

current_directory = os.getcwd()
print(current_directory)
file_path = os.path.join(current_directory, 'dtc.pickle')
file_path2 = os.path.join(current_directory, 'dtc_output.pickle')
df = pd.read_csv('cause_of_deaths.csv')

# Add filters for 'Year' and 'Country/Territory'
selected_year = st.selectbox(label='Select Year', options=df['Year'].unique())
selected_country = st.selectbox(label='Select Country/Territory', options=df['Country/Territory'].unique())

# Filter the dataframe based on selected year and country
filtered_df = df[(df['Year'] == selected_year) & (df['Country/Territory'] == selected_country)]

# Debugging statement to check the contents of filtered_df
print("Filtered Dataframe:")
print(filtered_df.head())

# Drop unnecessary columns
filtered_df.drop(columns=['Country/Territory', 'Code', 'Year'], inplace=True)

select = st.selectbox(label='Select Feature', options=filtered_df.columns)
submit = st.button(label='Submit')

if submit:
    if filtered_df.empty:
        st.warning("Insufficient data for model training. Please adjust your filters.")
    else:
        output, uniques = pd.factorize(filtered_df[select])
        cat_features = filtered_df.copy()
        num_features = filtered_df.drop(columns=[select])

        encoders = {}
        for feature in cat_features:
            if feature == select:
                continue
            encoder = LabelEncoder()
            encoded_values = encoder.fit_transform(cat_features[feature])
            cat_features.loc[:, feature] = encoded_values
            encoders[feature] = encoder

        output, uniques = pd.factorize(output)
        num_features = pd.get_dummies(num_features)
        features = pd.concat([cat_features, num_features], axis=1)

        x_train, x_test, y_train, y_test = train_test_split(features, output, test_size=0.3, random_state=42)

        decision_tree_model = DecisionTreeClassifier(criterion='entropy', max_depth=12, min_samples_leaf=1, min_samples_split=3)
        decision_tree_model.fit(x_train, y_train)
        y_train_pred = decision_tree_model.predict(x_train)
        train_accuracy = accuracy_score(y_train_pred, y_train)
        train_f1 = f1_score(y_train_pred, y_train, average='weighted')

        st.write("Training Accuracy:", train_accuracy)

        with open(file_path, 'wb') as rf_pickle:
            pickle.dump(decision_tree_model, rf_pickle)
            rf_pickle.close()

        with open(file_path2, 'wb') as output_pickle:
            pickle.dump(uniques, output_pickle)
            output_pickle.close()

        fig, ax = plt.subplots() 
        ax = sns.barplot(x=decision_tree_model.feature_importances_, y=features.columns) 
        plt.title('Important Features that could predict user subscription') 
        plt.xlabel('Importance') 
        plt.ylabel('Feature') 
        plt.tight_layout() 

        fig.savefig('dtc_feature_importance.png') 
        
        st.image('dtc_feature_importance.png')
