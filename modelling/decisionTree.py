import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
import pickle 
from sklearn.model_selection import train_test_split 
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import LabelEncoder
import streamlit as st

current_directory = os.getcwd()
print(current_directory)
file_path = os.path.join(current_directory, 'dtc.pickle')
file_path2 = os.path.join(current_directory, 'dtc_output.pickle')
df = pd.read_csv('cause_of_deaths.csv')
disease = df.drop(columns=['Country/Territory', 'Code', 'Year'])
select = st.selectbox(label='Select', options=disease.columns)
submit = st.button(label='submit')

if submit:
    output, uniques = pd.factorize(df[select])
    cat_features = df[['Country/Territory', 'Code', 'Year']]
    num_features = df.drop(columns=['Country/Territory', 'Code', 'Year', select])
    
    encoders = {}
    for feature in cat_features:
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
    train_precision = precision_score(y_train_pred, y_train, average='weighted')
    train_recall = recall_score(y_train_pred, y_train, average='weighted')
    train_f1 = f1_score(y_train_pred, y_train, average='weighted')

    print("Training Accuracy:", train_accuracy)
    print("Training Precision:", train_precision)
    print("Training Recall:", train_recall)
    print("Training F1 Score:", train_f1)

    st.write("Training Accuracy:", train_accuracy)
    st.write("Training Precision:", train_precision)
    st.write("Training Recall:", train_recall)
    st.write("Training F1 Score:", train_f1)

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
