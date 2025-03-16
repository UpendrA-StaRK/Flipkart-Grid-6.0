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
st.set_page_config(page_title="AI-Powered Size Chart Generator", layout="wide", page_icon="👕")

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

    # Use columns for arranging input sliders
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        height = st.slider('Height (cm)', 150, 190, 170)
    with col2:
        weight = st.slider('Weight (kg)', 50, 100, 70)
    with col3:
        chest = st.slider('Chest (cm)', 80, 120, 90)
    with col4:
        waist = st.slider('Waist (cm)', 60, 100, 75)
    with col5:
        hip = st.slider('Hip (cm)', 80, 120, 95)

    col6, col7 = st.columns(2)
    with col6:
        age = st.slider('Age', 18, 70, 30)
    with col7:
        gender = st.selectbox('Gender', ('Male', 'Female'))

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
    st.dataframe(data.head(), use_container_width=True)

    st.subheader('Your Input Parameters')
    st.table(input_df.T.rename(columns={0: 'Value'}))  # Corrected line

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

    if st.button('Find Similar Users'):
        st.session_state.similar_users = find_similar_users(input_df, data)
    else:
        if st.session_state.similar_users.empty:
            st.info("Click the button above to find similar users")

    if not st.session_state.similar_users.empty:
        st.subheader('Most Similar Users in Our Database')
        st.dataframe(st.session_state.similar_users.style.format({'Similarity Score': '{:.2f}'}), use_container_width=True)

    # Visualization of user data
    st.subheader('Population Distribution Visualizations')
    
    tab1, tab2, tab3 = st.tabs(["Height/Weight", "Body Measurements", "Demographics"])
    
    with tab1:
        fig1 = px.scatter(data, x='Height (cm)', y='Weight (kg)', color='Size Purchased',
                         title='Height vs Weight Distribution by Purchased Size',
                         labels={'Height (cm)': 'Height (cm)', 'Weight (kg)': 'Weight (kg)'})
        st.plotly_chart(fig1, use_container_width=True)
        
        fig2 = px.histogram(data, x='Height (cm)', nbins=30, 
                           title='Height Distribution',
                           labels={'Height (cm)': 'Height (cm)', 'count': 'Users'})
        st.plotly_chart(fig2, use_container_width=True)

    with tab2:
        fig3 = px.box(data, x='Gender', y='Chest (cm)', color='Gender',
                     title='Chest Size Distribution by Gender',
                     labels={'Gender': '', 'Chest (cm)': 'Chest (cm)'})
        st.plotly_chart(fig3, use_container_width=True)
        
        fig4 = px.violin(data, x='Size Purchased', y='Waist (cm)', color='Size Purchased',
                        title='Waist Size Distribution by Purchased Size')
        st.plotly_chart(fig4, use_container_width=True)

    with tab3:
        fig5 = px.pie(data, names='Gender', title='Gender Distribution')
        st.plotly_chart(fig5, use_container_width=True)
        
        fig6 = px.histogram(data, x='Age', nbins=20, 
                           title='Age Distribution',
                           labels={'Age': 'Age', 'count': 'Users'})
        st.plotly_chart(fig6, use_container_width=True)

elif selection == "Cluster Analysis":
    st.header('🛠️ Cluster Analysis')
    
    st.subheader('Clustering Parameters')
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
            model = AgglomerativeClustering(n_clusters=params.get('n_clusters', 5), 
                                           linkage=params.get('linkage', 'ward'))

        cluster_labels = model.fit_predict(scaled_features)
        data['Cluster'] = cluster_labels

        # Dimensionality reduction for visualization
        pca_data = PCA(n_components=2).fit_transform(scaled_features)
        data['PC1'], data['PC2'] = pca_data[:, 0], pca_data[:, 1]

        return data

    # Cluster data
    with st.spinner('Analyzing customer patterns...'):
        clustered_data = cluster_data(data, clustering_algorithm, **clustering_params)

    st.subheader('Cluster Insights')
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Cluster Distribution**")
        cluster_counts = clustered_data['Cluster'].value_counts().reset_index()
        cluster_counts.columns = ['Cluster', 'Count']
        fig = px.bar(cluster_counts, x='Cluster', y='Count', 
                    color='Cluster', text_auto=True,
                    labels={'Cluster': 'Cluster ID', 'Count': 'Customers'})
        st.plotly_chart(fig, use_container_width=True)
        
    with col2:
        st.markdown("**Cluster Characteristics**")
        numeric_cols = ['Height (cm)', 'Weight (kg)', 'Chest (cm)', 'Waist (cm)', 'Hip (cm)', 'Age']
        cluster_means = clustered_data.groupby('Cluster')[numeric_cols].mean().reset_index()
        st.dataframe(cluster_means.style.background_gradient(cmap='Blues'), 
                    use_container_width=True)

    st.subheader('Cluster Visualization')
    fig = px.scatter(clustered_data, x='PC1', y='PC2', color='Cluster',
                    hover_data=['Height (cm)', 'Weight (kg)', 'Size Purchased'],
                    title='Customer Clusters in 2D Space',
                    labels={'PC1': 'Principal Component 1', 'PC2': 'Principal Component 2'})
    st.plotly_chart(fig, use_container_width=True)

    st.subheader('Model Evaluation')
    try:
        silhouette_avg = silhouette_score(clustered_data[['PC1', 'PC2']], clustered_data['Cluster'])
        st.metric("Silhouette Score", f"{silhouette_avg:.2f}", 
                 help="Measures how similar objects are within clusters compared to other clusters (Higher is better)")
    except:
        st.warning("Could not calculate Silhouette Score for this clustering configuration")

elif selection == "Size Recommendations":
    st.header('📏 Smart Size Recommendations')
    
    if not st.session_state.similar_users.empty:
        st.subheader('Recommendations Based on Similar Customers')
        
        def recommend_size(similar_users):
            size_counts = similar_users['Size Purchased'].value_counts().reset_index()
            size_counts.columns = ['Size', 'Count']
            size_counts['Confidence'] = (size_counts['Count'] / size_counts['Count'].sum()) * 100
            return size_counts.sort_values('Confidence', ascending=False)
        
        size_recommendations = recommend_size(st.session_state.similar_users)
        
        col1, col2 = st.columns([1, 2])
        with col1:
            st.markdown("**Recommended Sizes**")
            st.dataframe(size_recommendations.style.format({'Confidence': '{:.1f}%'}),
                        use_container_width=True)
            
        with col2:
            st.markdown("**Recommendation Confidence**")
            fig = px.pie(size_recommendations, values='Confidence', names='Size',
                        hole=0.3, color_discrete_sequence=px.colors.sequential.Blues_r)
            fig.update_traces(textposition='inside', textinfo='percent+label')
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("---")
        st.subheader("Size Comparison Metrics")
        
        col3, col4, col5 = st.columns(3)
        with col3:
            st.metric("Most Common Size", size_recommendations.iloc[0]['Size'])
        with col4:
            st.metric("Success Probability", f"{size_recommendations.iloc[0]['Confidence']:.1f}%")
        with col5:
            return_rate = st.session_state.similar_users['Return'].mean() * 100
            st.metric("Historical Return Rate", f"{return_rate:.1f}%")
        
        st.markdown("---")
        st.subheader("Similar Customer Profiles")
        st.dataframe(st.session_state.similar_users[['Height (cm)', 'Weight (kg)', 
                                                    'Chest (cm)', 'Waist (cm)', 
                                                    'Size Purchased', 'Return']]
                    .style.format({'Similarity Score': '{:.2f}'}),
                    use_container_width=True)
        
    else:
        st.warning("No recommendation data available")
        st.info("Please complete the analysis in 'User Data Overview' first")

# Style the app with modern colors
st.markdown("""
    <style>
    .stApp {
        background-color: #FFFFFF;
        font-family: 'Inter', sans-serif;
    }
    .stSidebar {
        background-color: #2C3E50 !important;
    }
    .stSidebar .sidebar-content {
        color: #ECF0F1;
    }
    [data-testid="stHeader"] {
        background-color: #FFFFFF;
    }
    [data-testid="stToolbar"] {
        display: none;
    }
    .stTabs [aria-selected="true"] {
        background-color: #3498DB !important;
        color: white !important;
    }
    .st-bd {
        padding: 0.5rem;
    }
    .st-bb {
        border-bottom: 2px solid #3498DB;
    }
    .metric-container {
        border: 1px solid #BDC3C7;
        border-radius: 8px;
        padding: 1rem;
        margin: 0.5rem 0;
    }
    .stAlert {
        border-radius: 8px;
    }
    </style>
""", unsafe_allow_html=True)
