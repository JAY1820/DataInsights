import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
import os
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.metrics import silhouette_score, mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

# Load dataset
def load_data(file_path):
    """Load data from a CSV file."""
    try:
        data = pd.read_csv(file_path)
        print("Data loaded successfully.")
        return data
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

# Data cleaning
def clean_data(df):
    """Clean the dataset."""
    # Drop duplicate rows
    df = df.drop_duplicates()
    
    # Handle missing values
    df = df.fillna(method='ffill').fillna(method='bfill')
    
    # Convert date column to datetime
    df['date'] = pd.to_datetime(df['date'])
    
    # Remove any negative values
    df = df[df['sales'] >= 0]
    
    return df

# Feature engineering
def feature_engineering(df):
    """Create new features for the dataset."""
    df['day_of_week'] = df['date'].dt.dayofweek
    df['month'] = df['date'].dt.month
    df['quarter'] = df['date'].dt.quarter
    df['year'] = df['date'].dt.year
    
    # Create new features based on sales and revenue
    df['sales_per_day'] = df['sales'] / df['date'].dt.day
    df['revenue_per_day'] = df['revenue'] / df['date'].dt.day
    
    return df

# Data analysis
def analyze_data(df):
    """Analyze the dataset."""
    # Add a new column for year-month
    df['year_month'] = df['date'].dt.to_period('M')
    
    # Aggregate sales and revenue by year-month
    monthly_summary = df.groupby('year_month').agg({'sales': 'sum', 'revenue': 'sum'})
    
    # Calculate growth rates
    monthly_summary['sales_growth'] = monthly_summary['sales'].pct_change() * 100
    monthly_summary['revenue_growth'] = monthly_summary['revenue'].pct_change() * 100
    
    return monthly_summary

# Data visualization
def plot_data(df):
    """Create visualizations for the dataset."""
    plt.figure(figsize=(14, 7))
    
    # Sales and revenue over time
    plt.subplot(2, 1, 1)
    plt.plot(df.index, df['sales'], label='Sales', color='blue')
    plt.plot(df.index, df['revenue'], label='Revenue', color='green')
    plt.xlabel('Date')
    plt.ylabel('Amount')
    plt.title('Sales and Revenue Over Time')
    plt.legend()
    
    # Sales growth
    plt.subplot(2, 1, 2)
    plt.bar(df.index, df['sales_growth'], color='blue', alpha=0.6, label='Sales Growth')
    plt.bar(df.index, df['revenue_growth'], color='green', alpha=0.6, label='Revenue Growth')
    plt.xlabel('Date')
    plt.ylabel('Growth (%)')
    plt.title('Sales and Revenue Growth')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('sales_revenue_analysis.png')
    plt.show()

# Principal Component Analysis (PCA)
def perform_pca(df):
    """Perform PCA on the dataset."""
    features = ['sales', 'revenue']
    x = df.loc[:, features].values
    x = StandardScaler().fit_transform(x)
    
    pca = PCA(n_components=2)
    principal_components = pca.fit_transform(x)
    principal_df = pd.DataFrame(data=principal_components, columns=['principal_component_1', 'principal_component_2'])
    
    return principal_df

# K-Means Clustering
def kmeans_clustering(df):
    """Perform K-Means clustering on the dataset."""
    kmeans = KMeans(n_clusters=3)
    df['cluster'] = kmeans.fit_predict(df[['sales', 'revenue']])
    
    return df

# Image processing with OpenCV (dummy example)
def process_image(image_path):
    """Process an image with OpenCV."""
    try:
        img = cv2.imread(image_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 100, 200)
        
        # Save the processed image
        cv2.imwrite('processed_image.png', edges)
        print("Image processed and saved successfully.")
    except Exception as e:
        print(f"Error processing image: {e}")

# Train and evaluate models
def train_evaluate_models(df):
    """Train and evaluate machine learning models."""
    X = df[['sales', 'revenue']]
    y = df['sales']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Linear Regression
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    y_pred_lr = lr.predict(X_test)
    print(f"Linear Regression R2 Score: {r2_score(y_test, y_pred_lr)}")
    
    # Random Forest Regressor
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    y_pred_rf = rf.predict(X_test)
    print(f"Random Forest R2 Score: {r2_score(y_test, y_pred_rf)}")

def main():
    # File paths
    data_file = 'sales_data.csv'
    image_file = 'example_image.jpg'
    
    # Load and clean data
    data = load_data(data_file)
    if data is not None:
        data = clean_data(data)
        data = feature_engineering(data)
        
        # Analyze data
        summary = analyze_data(data)
        print(summary.head())
        
        # Visualize data
        plot_data(summary)
        
        # Perform PCA
        pca_df = perform_pca(data)
        print(pca_df.head())
        
        # Perform K-Means clustering
        clustered_data = kmeans_clustering(data)
        print(clustered_data.head())
        
        # Train and evaluate models
        train_evaluate_models(data)
        
        # Process image (if applicable)
        process_image(image_file)

if __name__ == '__main__':
    main()
