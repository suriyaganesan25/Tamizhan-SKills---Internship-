import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from mpl_toolkits.mplot3d import Axes3D

file_path = "customer_data.csv"
df = pd.read_csv(file_path)

print("First 5 rows of data:")
print(df.head())

features = ["Age", "Annual Income (k$)", "Spending Score (1-100)", "Purchase Frequency"]
X = df[features]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

kmeans = KMeans(n_clusters=4, random_state=42)
df["Cluster"] = kmeans.fit_predict(X_scaled)

print("\nCluster Centers (Scaled):")
print(kmeans.cluster_centers_)

plt.figure(figsize=(8, 6))
sns.scatterplot(data=df, x="Annual Income (k$)", y="Spending Score (1-100)", hue="Cluster", palette="viridis", s=100)
plt.title("Customer Segmentation (2D)")
plt.show()

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
scatter = ax.scatter(df["Age"], df["Annual Income (k$)"], df["Spending Score (1-100)"], 
                     c=df["Cluster"], cmap="viridis", s=50)
ax.set_xlabel("Age")
ax.set_ylabel("Annual Income (k$)")
ax.set_zlabel("Spending Score (1-100)")
plt.title("Customer Segmentation (3D)")
plt.show()

df.to_csv("segmented_customers.csv", index=False)
print("\nSegmented customer data saved as 'segmented_customers.csv'")
