import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import json
import urllib.request
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
import nltk
nltk.download('punkt')

# Page configuration
st.set_page_config(
    page_title="Industrial HR Geo-Visualization",
    page_icon="📊",
    layout="wide"
)

# Custom CSS
st.markdown("""
    <style>
    .stPlotlyChart {
        background-color: transparent !important;
    }
    .reportview-container {
        background: #f0f2f6;
    }
    div[data-testid="stMetricValue"] {
        font-size: 20px;
    }
    .cluster-terms {
        padding: 10px;
        background-color: #f0f2f6;
        border-radius: 5px;
        margin: 5px 0;
    }
    </style>
""", unsafe_allow_html=True)

@st.cache_data
def load_geojson():
    """Load India GeoJSON data"""
    url = "https://gist.githubusercontent.com/jbrobst/56c13bbbf9d97d187fea01ca62ea5112/raw/e388c4cae20aa53cb5090210a42ebb9b765c0a36/india_states.geojson"
    try:
        response = urllib.request.urlopen(url)
        return json.loads(response.read())
    except Exception as e:
        st.error(f"Error loading GeoJSON: {str(e)}")
        return None

@st.cache_data
def load_data():
    """Load and preprocess the dataset"""
    try:
        df = pd.read_csv(r"C:\ALL folder in dexstop\PycharmProjects\GUVI-Ai\Resource Management\cleaned_combined_data.csv", 
                        encoding='cp1252')
        # Print column names for debugging
        print("Available columns:", df.columns.tolist())
        return df
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None

def create_choropleth(data, feature_column):
    """Create choropleth map visualization"""
    try:
        geojson_data = load_geojson()
        if geojson_data is None:
            return None, None
        
        state_data = data.groupby('India/States', as_index=False).agg({
            feature_column: 'sum'
        })
        state_data[feature_column] = pd.to_numeric(state_data[feature_column], errors='coerce')
        
        # State name mapping
        state_name_mapping = {
            'STATE - WEST BENGAL': 'West Bengal',
            'STATE - RAJASTHAN': 'Rajasthan',
            'STATE - TAMIL NADU': 'Tamil Nadu',
            'STATE - MAHARASHTRA': 'Maharashtra',
            'STATE - UTTAR PRADESH': 'Uttar Pradesh',
            'STATE - BIHAR': 'Bihar',
            'STATE - MADHYA PRADESH': 'Madhya Pradesh',
            'STATE - ANDHRA PRADESH': 'Andhra Pradesh',
            'STATE - KARNATAKA': 'Karnataka',
            'STATE - GUJARAT': 'Gujarat',
            'STATE - ODISHA': 'Odisha',
            'STATE - KERALA': 'Kerala',
            'STATE - JHARKHAND': 'Jharkhand',
            'STATE - ASSAM': 'Assam',
            'STATE - PUNJAB': 'Punjab',
            'STATE - CHHATTISGARH': 'Chhattisgarh',
            'STATE - HARYANA': 'Haryana',
            'STATE - DELHI': 'NCT of Delhi',
            'STATE - JAMMU & KASHMIR': 'Jammu and Kashmir',
            'STATE - UTTARAKHAND': 'Uttarakhand',
            'STATE - HIMACHAL PRADESH': 'Himachal Pradesh',
            'STATE - TRIPURA': 'Tripura',
            'STATE - MEGHALAYA': 'Meghalaya',
            'STATE - MANIPUR': 'Manipur',
            'STATE - NAGALAND': 'Nagaland',
            'STATE - GOA': 'Goa',
            'STATE - ARUNACHAL PRADESH': 'Arunachal Pradesh',
            'STATE - PUDUCHERRY': 'Puducherry',
            'STATE - MIZORAM': 'Mizoram',
            'STATE - SIKKIM': 'Sikkim',
            'STATE - CHANDIGARH': 'Chandigarh',
            'STATE - DADRA & NAGAR HAVELI': 'Dadra and Nagar Haveli',
            'STATE - DAMAN & DIU': 'Daman and Diu',
            'STATE - LAKSHADWEEP': 'Lakshadweep',
            'STATE - ANDAMAN & NICOBAR': 'Andaman & Nicobar Islands'
        }
        
        state_data['mapped_state'] = state_data['India/States'].map(state_name_mapping)
        
        fig = px.choropleth(
            state_data,
            geojson=geojson_data,
            locations='mapped_state',
            featureidkey='properties.ST_NM',
            color=feature_column,
            color_continuous_scale=[
                [0, "#3B0284"],      # Deep Purple
                [0.2, "#2D5AA8"],    # Blue
                [0.4, "#1E9AB0"],    # Cyan
                [0.6, "#20B76B"],    # Green
                [0.8, "#7FD032"],    # Light Green
                [1, "#F6F926"]       # Yellow
            ],
            range_color=(state_data[feature_column].min(), state_data[feature_column].max()),
            scope="asia",
            hover_name='India/States',
            hover_data={feature_column: ':,.0f'},
            title=f'Distribution of {feature_column} across India'
        )

        fig.update_geos(
            visible=False,
            center=dict(lat=23.5937, lon=78.9629),
            projection_scale=4,
            showcoastlines=True,
            coastlinecolor="Black",
            showland=True,
            landcolor="white",
            fitbounds="locations",
            showcountries=True,
            countrycolor="Black",
            countrywidth=1,
            subunitcolor="Black",
            subunitwidth=1
        )

        fig.update_layout(
            margin={"r":0, "t":30, "l":0, "b":0},
            height=600,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            geo=dict(
                showframe=False,
                showcoastlines=True,
                projection_type='mercator',
                resolution=110,
                lonaxis_range=[65, 100],
                lataxis_range=[5, 40]
            )
        )
        
        return fig, state_data
    except Exception as e:
        print(f"Error in create_choropleth: {str(e)}")
        st.error(f"Error creating map: {str(e)}")
        return None, None

def create_gender_distribution(data, category):
    """Create pie chart for gender distribution"""
    try:
        # Map category to actual column names
        male_col = f"{category} -  Males"
        female_col = f"{category} -  Females"
        
        # Verify columns exist
        if male_col not in data.columns or female_col not in data.columns:
            print(f"Looking for columns: {male_col} and {female_col}")
            print("Available columns:", data.columns.tolist())
            st.error(f"Required columns not found: {male_col} or {female_col}")
            return None
        
        total_male = data[male_col].sum()
        total_female = data[female_col].sum()
        
        fig = px.pie(
            values=[total_male, total_female],
            names=['Male', 'Female'],
            title=f"Gender Distribution - {category}",
            color_discrete_sequence=['#2D5AA8', '#F6F926']
        )
        fig.update_traces(textinfo='percent+value')
        return fig
    except Exception as e:
        st.error(f"Error creating gender distribution: {str(e)}")
        print(f"Error details: {str(e)}")  # For debugging
        return None

def create_gender_bar_graph(data, category):
    """Create bar graph for gender comparison"""
    try:
        # Map category to actual column names
        male_col = f"{category} -  Males"
        female_col = f"{category} -  Females"
        
        # Verify columns exist
        if male_col not in data.columns or female_col not in data.columns:
            print(f"Looking for columns: {male_col} and {female_col}")
            print("Available columns:", data.columns.tolist())
            st.error(f"Required columns not found: {male_col} or {female_col}")
            return None
        
        male_count = data[male_col].sum()
        female_count = data[female_col].sum()
        
        # Create DataFrame for visualization
        gender_data = pd.DataFrame({
            'Category': [category, category],
            'Gender': ['Male', 'Female'],
            'Count': [male_count, female_count]
        })
        
        fig = px.bar(
            gender_data,
            x='Category',
            y='Count',
            color='Gender',
            title=f"Gender Comparison - {category}",
            color_discrete_sequence=['#2D5AA8', '#F6F926'],
            barmode='group'
        )
        
        fig.update_layout(
            xaxis_title=category,
            yaxis_title="Number of Workers",
            showlegend=True,
            height=400,
            xaxis={'tickangle': 45}
        )
        
        return fig
    except Exception as e:
        st.error(f"Error creating gender bar graph: {str(e)}")
        print(f"Error details: {str(e)}")  # For debugging
        return None

def create_distribution_histogram(data, selected_metric):
    """Create histogram distribution for selected metric"""
    try:
        # Create distribution plot
        fig = px.histogram(
            data,
            x=selected_metric,
            nbins=50,
            title=f"Distribution of {selected_metric}",
            color_discrete_sequence=['#2D5AA8']
        )
        
        fig.update_layout(
            xaxis_title=selected_metric,
            yaxis_title="Count",
            showlegend=False,
            height=400,
            bargap=0.1
        )
        
        # Add grid lines
        fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='gray')
        fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='gray')
        
        return fig
    except Exception as e:
        st.error(f"Error creating distribution histogram: {str(e)}")
        return None

def create_gender_pie_chart(data, category):
    """Create pie chart based on category selection"""
    try:
        # Map category to corresponding male and female columns
        category_mapping = {
            'Main Workers - Total-person': ('Main Workers - Total - Males', 'Main Workers - Total - Females'),
            'Main Workers - Rural - Persons': ('Main Workers - Rural - Males', 'Main Workers - Rural - Females'),
            'Main Workers - Urban - Persons': ('Main Workers - Urban - Males', 'Main Workers - Urban - Females'),
            'Marginal Workers - Total - Persons': ('Marginal Workers - Total - Males', 'Marginal Workers - Total - Females'),
            'Marginal Workers - Rural - Persons': ('Marginal Workers - Rural - Males', 'Marginal Workers - Rural - Females'),
            'Marginal Workers - Urban - Persons': ('Marginal Workers - Urban - Males', 'Marginal Workers - Urban - Females')
        }
        
        male_col, female_col = category_mapping[category]
        
        # Calculate totals
        total_male = data[male_col].sum()
        total_female = data[female_col].sum()
        
        # Create pie chart
        fig = px.pie(
            values=[total_male, total_female],
            names=['Male', 'Female'],
            title=f"Gender Distribution - {category}",
            color_discrete_sequence=['#2D5AA8', '#F6F926']
        )
        
        fig.update_traces(textinfo='percent+value')
        
        return fig
    except Exception as e:
        st.error(f"Error creating gender pie chart: {str(e)}")
        return None

def create_distribution_graph(data, selected_metric):
    """Create distribution graph"""
    try:
        industry_data = data.groupby('Industry')[selected_metric].sum().reset_index()
        industry_data = industry_data.sort_values(selected_metric, ascending=True).tail(15)
        
        fig = px.bar(
            industry_data,
            x=selected_metric,
            y='Industry',
            orientation='h',
            color=selected_metric,
            color_continuous_scale=['#3B0284', '#F6F926']
        )
        
        fig.update_layout(
            height=400,
            title=f"Top Industries by {selected_metric}",
            xaxis_title="Number of Workers",
            yaxis_title="Industry"
        )
        
        return fig
    except Exception as e:
        st.error(f"Error creating distribution graph: {str(e)}")
        return None

def create_industry_clusters(data):
    """Create industry clustering visualization"""
    try:
        tfidf = TfidfVectorizer(max_features=100)
        tfidf_matrix = tfidf.fit_transform(data['Industry'])
        
        n_clusters = 5
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        clusters = kmeans.fit_predict(tfidf_matrix)
        
        feature_names = tfidf.get_feature_names_out()
        cluster_terms = {}
        for i in range(n_clusters):
            top_indices = kmeans.cluster_centers_[i].argsort()[::-1][:5]
            top_terms = [feature_names[j] for j in top_indices]
            cluster_terms[f"Cluster {i+1}"] = top_terms
        
        cluster_sizes = pd.Series(clusters).value_counts()
        fig = px.bar(
            x=cluster_sizes.index,
            y=cluster_sizes.values,
            title="Industry Clusters Distribution",
            labels={'x': 'Cluster', 'y': 'Number of Industries'},
            color=cluster_sizes.values,
            color_continuous_scale=['#3B0284', '#F6F926']
        )
        
        return fig, cluster_terms
    except Exception as e:
        st.error(f"Error creating industry clusters: {str(e)}")
        return None, None

def main():
    st.title("Industrial Human Resource Geo-Visualization Dashboard")
    
    # Load data
    df = load_data()
    if df is None:
        return
    
    # Worker columns for visualization
    worker_columns = [
        "Main Workers - Total -  Persons",
        "Main Workers - Total - Males",
        "Main Workers - Total - Females",
        "Main Workers - Rural -  Persons",
        "Main Workers - Rural - Males",
        "Main Workers - Rural - Females",
        "Main Workers - Urban -  Persons",
        "Main Workers - Urban - Males",
        "Main Workers - Urban - Females",
        "Marginal Workers - Total -  Persons",
        "Marginal Workers - Total - Males",
        "Marginal Workers - Total - Females",
        "Marginal Workers - Rural -  Persons",
        "Marginal Workers - Rural - Males",
        "Marginal Workers - Rural - Females",
        "Marginal Workers - Urban -  Persons",
        "Marginal Workers - Urban - Males",
        "Marginal Workers - Urban - Females"
    ]
    
    # Tabs for different visualizations
    tab1, tab2, tab3 = st.tabs([
        "Geographical Distribution",
        "Gender & Distribution Analysis",
        "Industry Analysis"
    ])
    
    with tab1:
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.header("Geographical Distribution")
            selected_metric = st.selectbox(
                "Select metric to visualize:",
                options=worker_columns,
                key="geo_metric"
            )
            
            if selected_metric in df.columns:
                df[selected_metric] = pd.to_numeric(df[selected_metric], errors='coerce')
                fig, state_data = create_choropleth(df, selected_metric)
                if fig is not None:
                    st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            if state_data is not None:
                st.header("State Statistics")
                st.subheader("Top 5 States")
                top_states = state_data.nlargest(5, selected_metric)[['India/States', selected_metric]]
                top_states.columns = ['State', 'Workers']
                st.table(top_states)
                
                total_workers = state_data[selected_metric].sum()
                st.metric("Total Workers", f"{total_workers:,.0f}")
                
                avg_workers = state_data[selected_metric].mean()
                st.metric("Average per State", f"{avg_workers:,.0f}")
    
    with tab2:
        st.header("Distribution Analysis")
        
        # First dropdown for distribution histogram
        distribution_metrics = [
            "Main Workers - Total -  Persons",
            "Main Workers - Rural -  Persons",
            "Main Workers - Urban -  Persons",
            "Marginal Workers - Total -  Persons",
            "Marginal Workers - Rural -  Persons",
            "Marginal Workers - Urban -  Persons"
        ]
        
        selected_metric = st.selectbox(
            "Select metric for distribution analysis:",
            options=distribution_metrics,
            key="distribution_metric"
        )
        
        # Create and display distribution histogram
        dist_fig = create_distribution_histogram(df, selected_metric)
        if dist_fig is not None:
            st.plotly_chart(dist_fig, use_container_width=True)
        
        # Second dropdown for gender pie chart
        st.subheader("Gender Analysis")
        gender_categories = [
            'Main Workers - Total-person',
            'Main Workers - Rural - Persons',
            'Main Workers - Urban - Persons',
            'Marginal Workers - Total - Persons',
            'Marginal Workers - Rural - Persons',
            'Marginal Workers - Urban - Persons'
        ]
        
        selected_category = st.selectbox(
            "Select category for gender analysis:",
            options=gender_categories,
            key="gender_category"
        )
        
        # Create and display gender pie chart
        gender_fig = create_gender_pie_chart(df, selected_category)
        if gender_fig is not None:
            st.plotly_chart(gender_fig, use_container_width=True)
    
    with tab3:
        st.header("Industry Analysis")
        
        cluster_fig, cluster_terms = create_industry_clusters(df)
        if cluster_fig is not None:
            st.plotly_chart(cluster_fig, use_container_width=True)
        
        st.subheader("Industry Clusters Analysis")
        
        # Add cluster descriptions
        st.markdown("**Cluster 1**")
        st.markdown("Focused on: Sale, Retail, and Related Activities.")
        st.markdown("---")

        st.markdown("**Cluster 2**")
        st.markdown("Focused on: Manufacturing of Products and Other Related Industries.")
        st.markdown("---")

        st.markdown("**Cluster 3**")
        st.markdown("Focused on: Education, General, Technical, and Other Services.")
        st.markdown("---")

        st.markdown("**Cluster 4**")
        st.markdown("Focused on: Crop and Animal Production and Related Activities.")
        st.markdown("---")

        st.markdown("**Cluster 5**")
        st.markdown("Focused on: Service Activities and Other General Operations.")
        st.markdown("---")

        # Uncomment to display original cluster terms (optional)
        # if cluster_terms:
        #     for cluster, terms in cluster_terms.items():
        #         st.markdown(f"**{cluster}**")
        #         st.markdown(" • ".join(terms))
        #         st.markdown("---")
        
        st.subheader("Industry Distribution")
        selected_metric = st.selectbox(
            "Select metric to analyze:",
            options=worker_columns,
            key="industry_metric"
        )
        
        industry_dist_fig = create_distribution_graph(df, selected_metric)
        if industry_dist_fig is not None:
            st.plotly_chart(industry_dist_fig, use_container_width=True)

if __name__ == "__main__":
    main()
