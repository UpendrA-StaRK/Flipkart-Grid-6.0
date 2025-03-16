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
st.set_page_config(
    page_title="AI-Powered Size Chart Generator",
    layout="wide",
    page_icon="👕",
    initial_sidebar_state="expanded"
)

# Modern UI styling (Dark Mode)
st.markdown("""
    <style>
    /* General Background */
    .stApp {
        background: linear-gradient(135deg, #1f2937, #111827);
        font-family: 'Inter', sans-serif;
        color: #e5e7eb !important;
    }
    
    /* Sidebar Styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(195deg, #374151, #1f2937) !important;
        padding: 1rem;
    }
    .sidebar-content {
        color: #e5e7eb !important;
    }
    [data-baseweb="radio"] label {
        color: #e5e7eb !important;
    }

    /* Card Styling */
    .custom-card {
        background: rgba(31, 41, 55, 0.85);
        border-radius: 12px;
        padding: 2rem;
        box-shadow: 0 4px 8px rgba(0,0,0,0.3);
        margin: 1rem 0;
        color: #f3f4f6 !important;
    }

    /* Button Styling */
    .stButton>button {
        background: linear-gradient(45deg, #3b82f6, #1d4ed8) !important;
        color: white !important;
        border-radius: 8px !important;
        padding: 0.5rem 1.5rem !important;
        transition: all 0.3s ease-in-out !important;
        border: none !important;
        font-weight: bold;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 6px rgba(59,130,246,0.4);
    }

    /* Slider Styling */
    .stSlider>div>div>div>div {
        background: #3b82f6 !important;
    }

    /* Tabs Styling */
    [data-baseweb="tab-list"] button {
        padding: 0.75rem 1.5rem !important;
        border-radius: 8px !important;
        transition: all 0.3s ease-in-out !important;
        font-weight: bold;
        color: #e5e7eb !important;
    }
    [data-baseweb="tab-list"] button[aria-selected="true"] {
        background: #3b82f6 !important;
        color: white !important;
    }

    /* DataFrame Styling */
    .stDataFrame {
        border-radius: 12px !important;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3) !important;
    }

    /* Table Styling */
    table {
        background: rgba(31, 41, 55, 0.85) !important;
        border-radius: 12px !important;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3) !important;
        color: #e5e7eb !important;
    }
    th {
        background: #3b82f6 !important;
        color: white !important;
    }
    td {
        background: transparent !important;
        color: #f3f4f6 !important;
    }

    /* Section Headers */
    h2 {
        border-bottom: 3px solid #3b82f6;
        padding-bottom: 0.5rem !important;
        color: #f3f4f6 !important;
    }

    /* Plotly Chart Styling */
    .js-plotly-plot .plotly, .js-plotly-plot .plotly div {
        border-radius: 12px !important;
    }

    /* Gradient Effect for Title */
    .gradient-text {
        background: linear-gradient(45deg, #3b82f6, #1e3a8a);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 700 !important;
        font-family: 'Inter', sans-serif;
    }
    </style>
""", unsafe_allow_html=True)

# Title with gradient effect
st.markdown('<h1 class="gradient-text">🛍️ AI-Powered Size Chart Generator for Apparel Sellers</h1>', unsafe_allow_html=True)

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

    with st.container():
        st.markdown('<div class="custom-card">', unsafe_allow_html=True)
        
        col1, col2, col3, col4, col5 = st.columns(5)
        height = col1.slider('Height (cm)', 150, 190, 170)
        weight = col2.slider('Weight (kg)', 50, 100, 70)
        chest = col3.slider('Chest (cm)', 80, 120, 90)
        waist = col4.slider('Waist (cm)', 60, 100, 75)
        hip = col5.slider('Hip (cm)', 80, 120, 95)

        col6, col7 = st.columns(2)
        age = col6.slider('Age', 18, 70, 30)
        gender = col7.selectbox('Gender', ('Male', 'Female'))

        input_df = pd.DataFrame({
            'Height (cm)': [height],
            'Weight (kg)': [weight],
            'Chest (cm)': [chest],
            'Waist (cm)': [waist],
            'Hip (cm)': [hip],
            'Age': [age],
            'Gender': [gender]
        })

        st.markdown('</div>', unsafe_allow_html=True)

    st.subheader("Dataset Preview")
    st.dataframe(data.head(), use_container_width=True)

    st.subheader('User Input Parameters')
    st.dataframe(input_df, use_container_width=True)

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

    if st.button('Find Similar Users', key='find_similar'):
        st.session_state.similar_users = find_similar_users(input_df, data)

    st.subheader('Similar Users')
    st.dataframe(st.session_state.similar_users, use_container_width=True)

    st.subheader('User Data Visualization')
    with st.container():
        st.markdown('<div class="custom-card">', unsafe_allow_html=True)
        
        fig_height = px.histogram(data, x='Height (cm)', nbins=30, title='Height Distribution')
        st.plotly_chart(fig_height, use_container_width=True)
        
        fig_scatter = px.scatter(data, x='Height (cm)', y='Weight (kg)', 
                               color='Size Purchased', title='Height vs Weight')
        st.plotly_chart(fig_scatter, use_container_width=True)
        
        fig_box = px.box(data, x='Gender', y='Chest (cm)', 
                       color='Gender', title='Chest Size Distribution by Gender')
        st.plotly_chart(fig_box, use_container_width=True)
        
        st.markdown('</div>', unsafe_allow_html=True)

elif selection == "Cluster Analysis":
    st.header('🛠️ Cluster Analysis')

    with st.container():
        st.markdown('<div class="custom-card">', unsafe_allow_html=True)
        
        clustering_algorithm = st.selectbox(
            "Select Clustering Algorithm",
            ("KMeans", "DBSCAN", "Agglomerative")
        )

        if clustering_algorithm == "KMeans":
            n_clusters = st.slider("Number of Clusters", 2, 10, 5)
            clustering_params = {'n_clusters': n_clusters}
        elif clustering_algorithm == "DBSCAN":
            eps = st.slider("Epsilon", 0.1, 2.0, 0.5)
            min_samples = st.slider("Min Samples", 2, 10, 5)
            clustering_params = {'eps': eps, 'min_samples': min_samples}
        else:
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
            else:
                model = AgglomerativeClustering(n_clusters=params.get('n_clusters', 5), 
                                              linkage=params.get('linkage', 'ward'))

            cluster_labels = model.fit_predict(scaled_features)
            data['Cluster'] = cluster_labels

            pca_data = PCA(n_components=2).fit_transform(scaled_features)
            data['PC1'], data['PC2'] = pca_data[:, 0], pca_data[:, 1]
            return data

        with st.spinner('Performing clustering...'):
            clustered_data = cluster_data(data, clustering_algorithm, **clustering_params)

        st.subheader('Clustered Data')
        st.dataframe(clustered_data.head(), use_container_width=True)

        st.subheader('Cluster Distribution')
        cluster_counts = clustered_data['Cluster'].value_counts().reset_index()
        cluster_counts.columns = ['Cluster', 'Count']
        fig = px.bar(cluster_counts, x='Cluster', y='Count', title='Cluster Distribution')
        st.plotly_chart(fig, use_container_width=True)

        st.subheader('Cluster Visualization')
        fig = px.scatter(clustered_data, x='PC1', y='PC2', color='Cluster', title='Cluster Visualization')
        st.plotly_chart(fig, use_container_width=True)

        st.subheader('Clustering Evaluation')
        try:
            silhouette_avg = silhouette_score(clustered_data[['PC1', 'PC2']], clustered_data['Cluster'])
            st.metric("Silhouette Score", f"{silhouette_avg:.2f}")
        except:
            st.warning("Could not calculate Silhouette Score")
        
        st.markdown('</div>', unsafe_allow_html=True)

elif selection == "Size Recommendations":
    st.header('📏 Size Recommendations')
    
    with st.container():
        st.markdown('<div class="custom-card">', unsafe_allow_html=True)
        
        if st.session_state.similar_users.empty:
            st.warning("Please go to 'User Data Overview' tab and define your input parameters to get similar users.")
        else:
            def recommend_size(similar_users):
                size_counts = similar_users['Size Purchased'].value_counts().reset_index()
                size_counts.columns = ['Size', 'Count']
                size_counts['Confidence'] = size_counts['Count'] / size_counts['Count'].sum()
                return size_counts
            
            size_recommendations = recommend_size(st.session_state.similar_users)
            
            col1, col2 = st.columns([1, 2])
            with col1:
                st.subheader('Recommended Sizes')
                st.dataframe(size_recommendations, use_container_width=True)
            
            with col2:
                st.subheader('Confidence Distribution')
                fig = px.pie(size_recommendations, values='Confidence', 
                           names='Size', title='Size Recommendation Confidence')
                st.plotly_chart(fig, use_container_width=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
