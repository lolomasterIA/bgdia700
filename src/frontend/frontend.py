"""
Frontend va fournir l'ensemble des méthodes permettant l'affichage
"""

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


### Travaux sur les ingrédients ###
def display_kmeans_ingredient(df, k):
    # Affichage des clusters sous forme de graphique
    fig, ax = plt.subplots()
    scatter = ax.scatter(df['PC1'], df['PC2'],
                         c=df['Cluster'], cmap='viridis', alpha=0.6)
    for i, txt in enumerate(df['Ingrédient']):
        ax.annotate(txt, (df['PC1'][i], df['PC2'][i]), fontsize=8)

    legend1 = ax.legend(*scatter.legend_elements(), title="Clusters")
    ax.add_artist(legend1)
    plt.xlabel('Composante Principale 1')
    plt.ylabel('Composante Principale 2')
    plt.title(f"Clustering KMeans avec k = {k}")
    print("mon k=" + str(k))
    st.pyplot(fig)
