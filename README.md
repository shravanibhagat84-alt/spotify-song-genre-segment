# spotify-song-genre-segment
# Step 1: Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# Step 2: Load Dataset
df = pd.read_csv("/content/dataset.csv (1).zip")
print("Data Loaded Successfully")
print(df.head())

# Step 3: Basic Information
print("\nDataset Info:")
print(df.info())
print("\nMissing values per column:")
print(df.isnull().sum())

# Step 4: Select Important Numeric Features for Clustering
features = ['danceability', 'energy', 'loudness', 'speechiness',
'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo', 'duration_ms']

spotify_features = df[features]

# Drop missing values (if any)
spotify_features = spotify_features.dropna()

# Step 5: Scale the Data
scaler = StandardScaler()
scaled_data = scaler.fit_transform(spotify_features)

# Step 6: Visualize Data
plt.figure(figsize=(10,6))
sns.heatmap(spotify_features.corr(), cmap="coolwarm", annot=True)
plt.title(" Correlation Matrix of Audio Features")
plt.show()

# Histograms of all features
spotify_features.hist(figsize=(12,10), bins=20, color='skyblue')
plt.suptitle("Feature Distributions")
plt.show()

# Step 7: Apply K-Means Clustering
k = 4# you can change this number for different clusters
kmeans = KMeans(n_clusters=k, random_state=42)
df['Cluster'] = kmeans.fit_predict(scaled_data)

# Step 8: Visualize Clusters (using two features)
plt.figure(figsize=(8,6))
sns.scatterplot(x=df['danceability'], y=df['energy'], hue=df['Cluster'], palette='Set2')
plt.title(" Song Clusters Based on Danceability and Energy")
plt.show()

# Step 9: Analyze Each Cluster
cluster_summary = df.groupby('Cluster')[features].mean()
print("\nCluster Summary (Average feature values):")
print(cluster_summary)

# Step 10: Example – Songs from each Cluster
for i in range(k):
  print(f"\n Sample Songs in Cluster {i}:")
  print(df[df['Cluster'] == i][['track_name', 'artists', 'track_genre']].head(3))

print("\n Spotify Genre Segmentation Project Completed Successfully!")
