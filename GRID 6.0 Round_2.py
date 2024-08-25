import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

# Set Page Config
st.set_page_config(page_title="AI-Powered Size Chart Generator", layout="wide")

# Title
st.title("🛍️ AI-Powered Size Chart Generator for Apparel Sellers")

# Load the synthetic dataset
@st.cache_data
def load_data():
    np.random.seed(42)
    n_samples = 1000
    data = pd.DataFrame({
        'Height (cm)': np.random.uniform(150, 190, n_samples),
        'Weight (kg)': np.random.uniform(50, 100, n_samples),
        'Chest (cm)': np.random.uniform(80, 120, n_samples),
        'Waist (cm)': np.random.uniform(60, 100, n_samples),
        'Hip (cm)': np.random.uniform(80, 120, n_samples),
        'Age': np.random.randint(18, 70, n_samples),
        'Gender': np.random.choice(['Male', 'Female'], n_samples),
        'Size Purchased': np.random.choice(['XS', 'S', 'M', 'L', 'XL', 'XXL'], n_samples),
        'Return': np.random.choice([True, False], n_samples, p=[0.2, 0.8])
    })
    return data

data = load_data()

# Sidebar for navigation
st.sidebar.title("Navigation")
selection = st.sidebar.radio("Go to", ["User Data Overview", "Cluster Analysis", "Size Recommendations"])

# Store similar_users in session state
if 'similar_users' not in st.session_state:
    st.session_state.similar_users = pd.DataFrame()

# Input Parameters at the Top
if selection == "User Data Overview":
    st.header("🔍 User Input Parameters")

    # Use columns for arranging input sliders at the top
    col1, col2, col3, col4, col5 = st.columns(5)

    height = col1.slider('Height (cm)', 150, 190, 170)
    weight = col2.slider('Weight (kg)', 50, 100, 70)
    chest = col3.slider('Chest (cm)', 80, 120, 90)
    waist = col4.slider('Waist (cm)', 60, 100, 75)
    hip = col5.slider('Hip (cm)', 80, 120, 95)

    col6, col7 = st.columns(2)
    age = col6.slider('Age', 18, 70, 30)
    gender = col7.selectbox('Gender', ('Male', 'Female'))

    # Combine input data into a DataFrame
    input_df = pd.DataFrame({
        'Height (cm)': [height],
        'Weight (kg)': [weight],
        'Chest (cm)': [chest],
        'Waist (cm)': [waist],
        'Hip (cm)': [hip],
        'Age': [age],
        'Gender': [gender]
    })

    st.subheader("Dataset Preview")
    st.write(data.head())

    st.subheader('User Input Parameters')
    st.write(input_df)

    def find_similar_users(input_data, dataset, n_neighbors=5):
        features = ['Height (cm)', 'Weight (kg)', 'Chest (cm)', 'Waist (cm)', 'Hip (cm)']
        scaler = StandardScaler()
        dataset_scaled = scaler.fit_transform(dataset[features])
        input_scaled = scaler.transform(input_data[features])
        nbrs = NearestNeighbors(n_neighbors=n_neighbors, metric='euclidean').fit(dataset_scaled)
        distances, indices = nbrs.kneighbors(input_scaled)
        similar_users = dataset.iloc[indices[0]]
        similar_users['Similarity Score'] = 1 / (1 + distances[0])
        return similar_users

    # Find similar users and store in session state
    st.session_state.similar_users = find_similar_users(input_df, data)

    st.subheader('Similar Users')
    st.write(st.session_state.similar_users)

    # Visualization of user data
    st.subheader('User Data Visualization')

    # Histogram of height
    fig_height = px.histogram(data, x='Height (cm)', nbins=30, title='Height Distribution')
    st.plotly_chart(fig_height)

    # Scatter plot of height vs weight
    fig_scatter = px.scatter(data, x='Height (cm)', y='Weight (kg)', color='Size Purchased', title='Height vs Weight')
    st.plotly_chart(fig_scatter)

    # Box plot of chest sizes by gender
    fig_box = px.box(data, x='Gender', y='Chest (cm)', color='Gender', title='Chest Size Distribution by Gender')
    st.plotly_chart(fig_box)

elif selection == "Cluster Analysis":
    st.header('🛠️ Cluster Analysis')

    # Clustering Algorithm Selection
    clustering_algorithm = st.selectbox(
        "Select Clustering Algorithm",
        ("KMeans", "DBSCAN", "Agglomerative")
    )

    # Clustering Parameters
    if clustering_algorithm == "KMeans":
        n_clusters = st.slider("Number of Clusters", 2, 10, 5)
        clustering_params = {'n_clusters': n_clusters}
    elif clustering_algorithm == "DBSCAN":
        eps = st.slider("Epsilon", 0.1, 2.0, 0.5)
        min_samples = st.slider("Min Samples", 2, 10, 5)
        clustering_params = {'eps': eps, 'min_samples': min_samples}
    else:  # Agglomerative
        n_clusters = st.slider("Number of Clusters", 2, 10, 5)
        linkage = st.selectbox("Linkage", ("ward", "complete", "average"))
        clustering_params = {'n_clusters': n_clusters, 'linkage': linkage}

    def cluster_data(data, algorithm, **params):
        features = ['Height (cm)', 'Weight (kg)', 'Chest (cm)', 'Waist (cm)', 'Hip (cm)']
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(data[features])

        if algorithm == "KMeans":
            model = KMeans(n_clusters=params.get('n_clusters', 5), random_state=42)
        elif algorithm == "DBSCAN":
            model = DBSCAN(eps=params.get('eps', 0.5), min_samples=params.get('min_samples', 5))
        else:  # Agglomerative
            model = AgglomerativeClustering(n_clusters=params.get('n_clusters', 5), linkage=params.get('linkage', 'ward'))

        cluster_labels = model.fit_predict(scaled_features)
        data['Cluster'] = cluster_labels

        # Dimensionality reduction for visualization
        pca_data = PCA(n_components=2).fit_transform(scaled_features)
        data['PC1'], data['PC2'] = pca_data[:, 0], pca_data[:, 1]

        return data

    # Cluster data
    with st.spinner('Performing clustering...'):
        clustered_data = cluster_data(data, clustering_algorithm, **clustering_params)

    st.subheader('Clustered Data')
    st.write(clustered_data.head())

    st.subheader('Cluster Distribution')
    cluster_counts = clustered_data['Cluster'].value_counts().reset_index()
    cluster_counts.columns = ['Cluster', 'Count']
    fig = px.bar(cluster_counts, x='Cluster', y='Count', title='Cluster Distribution')
    st.plotly_chart(fig)

    st.subheader('Cluster Visualization')
    fig = px.scatter(clustered_data, x='PC1', y='PC2', color='Cluster', title='Cluster Visualization')
    st.plotly_chart(fig)

    st.subheader('Clustering Evaluation')
    silhouette_avg = silhouette_score(clustered_data[['PC1', 'PC2']], clustered_data['Cluster'])
    st.write(f'Silhouette Score: {silhouette_avg:.2f}')

elif selection == "Size Recommendations":
    st.header('📏 Size Recommendations')
    
    st.subheader('Recommended Size for Similar Users')
    
    # Define a default empty DataFrame for similar_users if not available
    if st.session_state.similar_users.empty:
        st.write("Please go to 'User Data Overview' tab and define your input parameters to get similar users.")
    else:
        def recommend_size(similar_users):
            size_counts = similar_users['Size Purchased'].value_counts().reset_index()
            size_counts.columns = ['Size', 'Count']
            size_counts['Confidence'] = size_counts['Count'] / size_counts['Count'].sum()
            return size_counts
        
        size_recommendations = recommend_size(st.session_state.similar_users)
        st.write(size_recommendations)

        fig = px.pie(size_recommendations, values='Confidence', names='Size', title='Size Recommendation Confidence')
        st.plotly_chart(fig)

# Style the app with aesthetic colors
st.markdown("""
    <style>
    .stApp {
        background-color: #303234; /* White Background */
        font-family: 'Arial', sans-serif;
        color: #303234; /* Black text */
    }
    .stSidebar {
        background-color: #303234; /* White Background */
    }
    .stSidebar .sidebar-content {
        color: #303234; /* Black text */
    }
    .stSidebar [data-baseweb="radio"] {
        color: #303234; /* Black text */
    }
    .stTabs [data-baseweb="tab-list"] button {
        color: #303234; /* Black text */
        background-color: #303234; /* Light Gray Background */
        border-bottom: 2px solid #303234; /* Black border */
    }
    .stTabs [data-baseweb="tab-list"] button [data-testid="stTabs-tab-active"] {
        color: #303234; /* White text */
        background-color: #303234; /* Black background */
        border-bottom: 2px solid #303234; /* Black border */
    }
    .stMarkdown h2, .stMarkdown h3 {
        color: #303234; /* Black text */
    }
    table {
        background-color: #303234; /* White Background for Tables */
        color: #303234; /* Black text */
    }
    th {
        background-color: #303234; /* Dark Gray for Table Headers */
        color: #303234; /* White text for headers */
    }
    td {
        background-color: #303234; /* White for Table Data */
        color: #303234; /* Black text */
    }
    </style>
""", unsafe_allow_html=True)
