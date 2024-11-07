import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler  # Re-added this import

# Load and clean data
data_path = r'C:\ALL folder in desktop\PycharmProjects\GUVI-Ai\Resource Management\cleaned_combined_data.csv'
data = pd.read_csv(data_path)

# Remove extra spaces from column names
data.columns = data.columns.str.replace(r'\s+', ' ', regex=True)

# Drop rows with missing 'Industry' values and fill NaNs in numeric columns with their column mean
data = data.dropna(subset=['Industry']).copy()
for col in data.select_dtypes(include=['float64', 'int64']).columns:
    data[col] = data[col].fillna(data[col].mean())

# Streamlit UI Components
st.title("Industrial Human Resource Geo-Visualization Dashboard")

# Section 1: State and District Selection
st.header("Select State and District")

# Select State
states = data['India/States'].unique()
selected_state = st.selectbox('Select State:', states)

# Filter data based on selected state
state_data = data[data['India/States'] == selected_state]

# Select district based on selected state
districts = state_data['District Code'].unique()
selected_district = st.selectbox('Select District:', districts)

# Filter data based on selected district
district_data = state_data[state_data['District Code'] == selected_district]

# Section 2: Worker Counts and Industry Information
st.header("Worker and Industry Information")

# Define required columns for worker counts
required_columns = [
    'Main Workers - Total - Persons',
    'Main Workers - Total - Males',
    'Main Workers - Total - Females',
    'Marginal Workers - Total - Persons',
    'Marginal Workers - Total - Males',
    'Marginal Workers - Total - Females'
]

# Check and display worker counts if available
missing_columns = [col for col in required_columns if col not in district_data.columns]

if not missing_columns:
    st.write("Worker Counts:")
    st.dataframe(district_data[required_columns])
    
    total_main_workers = district_data['Main Workers - Total - Persons'].sum()
    total_main_males = district_data['Main Workers - Total - Males'].sum()
    total_main_females = district_data['Main Workers - Total - Females'].sum()
    total_marginal_workers = district_data['Marginal Workers - Total - Persons'].sum()
    total_marginal_males = district_data['Marginal Workers - Total - Males'].sum()
    total_marginal_females = district_data['Marginal Workers - Total - Females'].sum()
    
    st.write("Total Workers Information:")
    st.write(f"Main Workers (Total): {total_main_workers}")
    st.write(f"Main Workers (Males): {total_main_males}")
    st.write(f"Main Workers (Females): {total_main_females}")
    st.write(f"Marginal Workers (Total): {total_marginal_workers}")
    st.write(f"Marginal Workers (Males): {total_marginal_males}")
    st.write(f"Marginal Workers (Females): {total_marginal_females}")
else:
    st.write(f"Missing columns for worker counts: {missing_columns}")

# Display industry information
if 'Industry' in district_data.columns:
    industry_info = district_data[['Industry', 'Class', 'Main Workers - Total - Persons']]
    st.write("Industry Information:")
    st.dataframe(industry_info)

# Section 3: Industry Clustering
st.header("Industry Clustering")
st.write("Applying TF-IDF and KMeans for Industry Clustering")

tfidf = TfidfVectorizer(stop_words='english', max_features=50)
industry_tfidf = tfidf.fit_transform(data['Industry'].astype(str))
kmeans = KMeans(n_clusters=5, random_state=42)
data['industry_cluster'] = kmeans.fit_predict(industry_tfidf)

# Section 4: Worker Distribution by Industry
st.header("Worker Distribution by Industry")
worker_type = st.selectbox("Select Worker Type", ["Main Workers - Urban - Persons", "Main Workers - Rural - Persons"])

if worker_type in data.columns:
    worker_distribution = data.groupby("Industry")[worker_type].sum().sort_values(ascending=False)
    st.bar_chart(worker_distribution)
else:
    st.error(f"Column '{worker_type}' not found in the data")

# Section 5: Feature Distribution
st.subheader("Feature Distribution")
scaler = StandardScaler()
numerical_cols = data.select_dtypes(include=['float64', 'int64']).columns
scaled_features = scaler.fit_transform(data[numerical_cols])
scaled_data = pd.DataFrame(scaled_features, columns=numerical_cols)

feature = st.selectbox("Select a feature to visualize", options=scaled_data.columns)
if feature in scaled_data.columns:
    fig = px.histogram(data, x=feature, nbins=30, title=f"Distribution of {feature}")
    st.plotly_chart(fig)

# Section 6: Correlation Matrix
st.subheader("Correlation Matrix")
if not scaled_data.empty:
    correlation_matrix = scaled_data.corr()
    fig = px.imshow(correlation_matrix, text_auto=True, aspect="auto", title="Correlation Matrix")
    st.plotly_chart(fig)

# Display cleaned data columns for debugging
st.write("Data Columns:", data.columns.tolist())
