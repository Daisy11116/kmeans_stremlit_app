#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 20 16:13:53 2025

@author: supatsarasaennang
"""

import streamlit as st
import pickle
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs

# Set page config — MUST be first Streamlit command
st.set_page_config(page_title="k-Means Clustering App", layout="centered")

# Set the title
st.title("K-means Clustering Visualizer by Supatsara Saennang")

# Load model
with open('kmeans_model.pkl', 'rb') as f:
    loaded_model = pickle.load(f)

# Load dataset
X, _ = make_blobs(n_samples=300, centers=loaded_model.n_clusters, cluster_std=0.60, random_state=0)

# Predict using the loaded model
y_kmeans = loaded_model.predict(X)ax.legend()

# Plotting
fig, ax = plt.subplots()
scatter = ax.scatter(X[:, 0], X[:, 1], c=y_kmeans, cmap='viridis')
ax.scatter(loaded_model.cluster_centers_[:, 0], loaded_model.cluster_centers_[:, 1], s=300, c='red', marker='X')
ax.set_title('k-Means Clustering')
ax.legend()
# Show the plot in Streamlit
st.pyplot(fig)
