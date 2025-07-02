import pandas as pd
from sklearn.cluster import KMeans
import plotly.express as px

# Load the data
data = pd.read_csv("/Users/macbookpro/Downloads/golomt bank intern/algorithms and codes/customers-data - customers-data.csv")

# Fit KMeans
k = 5
kmeans = KMeans(n_clusters=k, init='k-means++', max_iter=400, random_state=42)
features = data[['products_purchased','complains','money_spent']]
labels = kmeans.fit_predict(features)

# Store cluster labels
data["cluster"] = labels.astype(str)  # Make sure clusters are strings for category color

# 3D Scatter plot with Plotly Express

data["cluster"] = labels.astype(str)
fig = px.scatter_3d(
    data,
    x="products_purchased",
    y="complains",
    z="money_spent",
    color="cluster"
)

fig.update_layout(
    scene=dict(
        zaxis=dict(range=[0,5000]),
        xaxis = dict(range=[0,15])
    )
)

fig.show()

# Assuming your DataFrame is called data and has a "cluster" column as string labels
# For clarity, make sure cluster is int for aggregation:
data["cluster"] = data["cluster"].astype(int)

# Compute mean spending per cluster
cluster_means = data.groupby("cluster")["money_spent"].mean()
print(cluster_means)

# Get the cluster index with highest spending
highest_spending_cluster = cluster_means.idxmax()
print("Cluster with highest average spending:", highest_spending_cluster)

high_spenders = data[data["cluster"] == highest_spending_cluster]

# Display their IDs
print("Customer IDs in the highest spending cluster:")
print(high_spenders["customer_id"])

