import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split,cross_val_score  
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.metrics import r2_score, mean_absolute_error, accuracy_score, precision_score, recall_score, f1_score
from sklearn.cluster import KMeans, AgglomerativeClustering,DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA

death_data = pd.read_csv('cause_of_deaths.csv')

# Sidebar to select page
page = st.sidebar.selectbox("Select Page", ["Main Dashboard", "KNN", "Naive Bayes", "Decision"])

# Sum all cause of deaths
cause_of_deaths = [col for col in death_data.columns if col not in ('Country/Territory', 'Code', 'Year')]
death_data['Total Deaths'] = death_data[cause_of_deaths].sum(axis=1)

# Interactive Main Dashboard
if page == "Main Dashboard":
    st.title("Health Statistics Dashboard")

    # Visualization 1: Total Deaths Over Time
    st.subheader("Total Deaths Over Time")
    year_range = st.slider("Select Year Range", int(death_data["Year"].min()), int(death_data["Year"].max()), (1990, 2019))
    filtered_data = death_data[(death_data["Year"] >= year_range[0]) & (death_data["Year"] <= year_range[1])]
    total_deaths_over_time = filtered_data.groupby("Year")[death_data.columns[3:]].sum().sum(axis=1)
    st.line_chart(total_deaths_over_time)

 # Visualization 2 - Total Death By Country
    st.title('Total Deaths by Country')
    selected_countries = st.selectbox('Select a Country', death_data['Country/Territory'].unique(), key='selected_countries_2')
    country_year_range = st.slider('Select Year Range', min_value=int(death_data['Year'].min()), max_value=int(death_data['Year'].max()), value=(1990, 2019), key='country_year_range_2')

    selected_country_data = death_data[death_data['Country/Territory'] == selected_countries]
    selected_country_data_filtered = selected_country_data[(selected_country_data['Year'] >= country_year_range[0]) & (selected_country_data['Year'] <= country_year_range[1])]

    st.subheader('Total Deaths Over Time')
    fig, ax = plt.subplots()
    chart = selected_country_data_filtered.groupby(['Year', 'Country/Territory'])['Total Deaths'].sum().unstack().plot.line(ax=ax)
    ax.set_xlabel('Year')
    ax.set_ylabel('Total Deaths')
    st.pyplot(fig)

    st.subheader(f'Total Deaths for {selected_countries} (within {country_year_range[0]} - {country_year_range[1]})')
    total_death = selected_country_data_filtered['Total Deaths'].sum()
    st.write(f'Total death for {selected_countries}: {total_death}')

    # Visualization 3 - Total Cases of Diseases
    st.title('Total Causes of Deaths')
    selected_diseases = st.multiselect('Select Causes', cause_of_deaths, key='selected_diseases_3')

    disease_totals = death_data[selected_diseases].sum().sort_values(ascending=True)

    disease_year_range = st.slider('Select Year Range', min_value=int(death_data['Year'].min()), max_value=int(death_data['Year'].max()), value=(1990, 2019), key='disease_year_range_3')
    disease_year_data = death_data[(death_data['Year'] >= disease_year_range[0]) & (death_data['Year'] <= disease_year_range[1])]

    st.subheader(f'Total Causes of Deaths (within {disease_year_range[0]} - {disease_year_range[1]})')
    fig, ax = plt.subplots()
    chart = disease_year_data.groupby('Year')[selected_diseases].sum().plot.line(ax=ax)
    ax.set_xlabel('Year')
    ax.set_ylabel('Total Cases')
    st.pyplot(fig)

    for disease in selected_diseases:
        st.write(f'{disease}: {disease_year_data[disease].sum()}')
                       
    # Visualization 4 - Top Causes of Death
    st.title('Top and Low Causes of Death')

    selected_country = st.selectbox('Select Country', death_data['Country/Territory'].unique(), index=0)
    range_selector = st.slider('Select Range', min_value=1, max_value=31, value=3)
    selected_country_data = death_data[death_data['Country/Territory'] == selected_country]

    country_cause_totals = selected_country_data[cause_of_deaths].sum()
    country_cause_totals_sorted = country_cause_totals.sort_values(ascending=False).head(range_selector)
    
    fig_top, ax_top = plt.subplots(figsize=(12, 6))
    ax_top.bar(country_cause_totals_sorted.index, country_cause_totals_sorted.values)
    ax_top.set_xlabel('Causes of Death')
    ax_top.set_ylabel('Total Deaths')  
    ax_top.set_xticklabels(ax_top.get_xticklabels(), rotation=45, ha='right')

    # Visualization 5 - Low Causes of Death
    country_cause_totals_low_sorted = country_cause_totals.sort_values(ascending=True).head(range_selector)

    fig_low, ax_low = plt.subplots(figsize=(12, 6)) 
    ax_low.bar(country_cause_totals_low_sorted.index, country_cause_totals_low_sorted.values)
    ax_low.set_xlabel('Causes of Death')
    ax_low.set_ylabel('Total Deaths')
    ax_low.set_xticklabels(ax_low.get_xticklabels(), rotation=45, ha='right') 
    
    col1, col2 = st.columns(2)

    with col1:
        st.pyplot(fig_top)
        st.subheader(f'Top {range_selector} Causes of Death in {selected_country}')
        for cause, total_deaths in country_cause_totals_sorted.items():
            st.write(f'- Cause: {cause}, Total Deaths: {total_deaths}')

    with col2:
        st.pyplot(fig_low)
        st.subheader(f'Low {range_selector} Causes of Death in {selected_country}')
        for cause, total_deaths in country_cause_totals_low_sorted.items():
            st.write(f'- Cause: {cause}, Total Deaths: {total_deaths}')


    #visualization 6 - Comparison
    st.title('Country Comparison')

    country_1 = st.selectbox('Select Country 1', death_data['Country/Territory'].unique())
    country_2 = st.selectbox('Select Country 2', death_data['Country/Territory'].unique())

    comp_year_range = st.slider('Select Year Range', min_value=int(death_data['Year'].min()), max_value=int(death_data['Year'].max()), value=(1990, 2019))
    st.subheader(f'{country_1} VS {country_2}')

    comparison_data = death_data[(death_data['Country/Territory'].isin([country_1, country_2])) & (death_data['Year'] >= comp_year_range[0]) & (death_data['Year'] <= comp_year_range[1])]

    total_death_country1 = comparison_data[comparison_data['Country/Territory'] == country_1]['Total Deaths'].sum()
    st.write(f'Total death for {country_1} : {total_death_country1}')
    total_death_country2 = comparison_data[comparison_data['Country/Territory'] == country_2]['Total Deaths'].sum()
    st.write(f'Total death for {country_2} : {total_death_country2}')

    fig, ax = plt.subplots()
    for country in [country_1, country_2]:
        country_data = comparison_data[comparison_data['Country/Territory'] == country]
        ax.scatter(country_data['Year'], country_data['Total Deaths'], label=country, marker='o')

    ax.set_xlabel('Year')
    ax.set_ylabel('Total Deaths')
    ax.legend()
    st.pyplot(fig)

    comp_selected_diseases = st.multiselect('Select causes', cause_of_deaths, default=cause_of_deaths)
    comp_data_selected_diseases = comparison_data[['Year', 'Country/Territory'] + comp_selected_diseases]

    for disease in comp_selected_diseases:
        st.subheader(f'Total Deaths for {disease} in {country_1} vs {country_2}')

        disease_data_country_1 = comp_data_selected_diseases[comp_data_selected_diseases['Country/Territory'] == country_1][['Year', disease]].rename(columns={disease: f'{country_1} {disease} deaths'})
        disease_data_country_2 = comp_data_selected_diseases[comp_data_selected_diseases['Country/Territory'] == country_2][['Year', disease]].rename(columns={disease: f'{country_2} {disease} deaths'})

        total_death_country1_selected_disease = disease_data_country_1[f'{country_1} {disease} deaths'].sum()
        total_death_country2_selected_disease = disease_data_country_2[f'{country_2} {disease} deaths'].sum()
        st.write(f'{country_1}, {disease} Total Deaths: {total_death_country1_selected_disease}')
        st.write(f'{country_2}, {disease} Total Deaths: {total_death_country2_selected_disease}')
        
        merged_data = pd.merge(disease_data_country_1, disease_data_country_2, on='Year', how='inner')

        fig, ax = plt.subplots()
        ax.plot(merged_data['Year'], merged_data[f'{country_1} {disease} deaths'], label=country_1)
        ax.plot(merged_data['Year'], merged_data[f'{country_2} {disease} deaths'], label=country_2)
        ax.set_xlabel('Year')
        ax.set_ylabel('Total Deaths')
        ax.legend()
        st.pyplot(fig)
