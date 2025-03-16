# AI-Powered Size Chart Generator for Apparel Sellers

## Introduction

The **AI-Powered Size Chart Generator** is an innovative tool that helps apparel sellers create accurate and personalized size charts using data-driven insights. By leveraging machine learning techniques and interactive visualizations, this application assists in understanding user measurements, clustering similar profiles, and recommending optimal apparel sizes.

## Features

- **Interactive User Input:** Collects key measurements like height, weight, chest, waist, and hip, along with demographic details.
- **Cluster Analysis:** Uses clustering techniques (KMeans, DBSCAN, Agglomerative) to segment users based on physical attributes and purchase history.
- **Size Recommendations:** Analyzes purchase trends among similar users to provide personalized size suggestions.
- **Dynamic Visualizations:** Utilizes Plotly and Streamlit for real-time, intuitive data representation.

## How It Works

### 1️⃣ User Data Overview
- **Input Measurements:** Users provide height, weight, chest, waist, hip, age, and gender using sliders and dropdowns.
- **Data Preview & Similarity Analysis:** Displays a synthetic dataset preview and finds similar users using a nearest neighbors algorithm, assigning a similarity score.

### 2️⃣ Cluster Analysis
- **Algorithm Selection:** Choose from KMeans, DBSCAN, or Agglomerative clustering.
- **Data Processing & Visualization:** The app scales data, applies PCA for dimensionality reduction, and visualizes clusters with evaluation metrics (e.g., silhouette score).

### 3️⃣ Size Recommendations
- **Data Aggregation:** Compiles size purchase trends from similar users.
- **Size Suggestion:** Provides a recommended size and visualizes confidence levels using interactive charts.

## Tech Stack

- **Backend & Processing:**
  - [Python](https://www.python.org/)
  - Libraries: [pandas](https://pandas.pydata.org/), [numpy](https://numpy.org/), [scikit-learn](https://scikit-learn.org/), [plotly](https://plotly.com/)

- **Frontend:**
  - [Streamlit](https://streamlit.io/) for interactive web applications

- **Deployment:**
  - Deployable via Streamlit Sharing, Heroku, or any Python-compatible cloud platform

## Installation and Setup

### Prerequisites

- [Python 3](https://www.python.org/) installed
- `pip` (Python package manager)

### Steps

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/UpendrA-StaRK/Flipkart-Grid-6.0.git
   cd Flipkart-Grid-6.0
   ```

2. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Application:**
   ```bash
   streamlit run main.py
   ```

## Contributions

Contributions are welcome! Feel free to fork the repository, create issues, or submit pull requests. Your feedback will help improve this tool for apparel sellers worldwide.

## Remarks

This project was developed as my solution to **Flipkart Grid SDE 6.0**'s problem statement. It was my competition submission and helped me become a **semifinalist**.

📌 **GitHub Repository:** [Flipkart-Grid-6.0](https://github.com/UpendrA-StaRK/Flipkart-Grid-6.0)

Happy coding! 🚀

