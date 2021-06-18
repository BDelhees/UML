# -*- coding: utf-8 -*-
"""
Created on Tue Jun  1 08:38:57 2021

@author: budde


UML 2: Clustering K-Means
"""

####K-Means Clustering
#Import basic libraries 

import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


#Import data

data_path = "..."

#PCA df
pca_df = pd.read_csv(f'{data_path}/pca_df.csv', index_col = 0)

#drop last 3 columns of pca_df to reduce dimensionality

pca_df = pca_df[['PC1', 'PC2']]

#Scaled df

scaled_df = pd.read_csv(f'{data_path}/scaled_df.csv', index_col = 0)




###K-Means Clustering

#libraries
from sklearn.cluster import KMeans, AgglomerativeClustering, AffinityPropagation
from sklearn.cluster import KMeans, AgglomerativeClustering, AffinityPropagation
from sklearn.cluster import MeanShift, estimate_bandwidth, SpectralClustering
from sklearn.cluster import DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score
import scipy.cluster.hierarchy as shc

#seed
seed = 123

#Model 1

km1 = KMeans(n_clusters = 2,
             init = 'random',
             max_iter = 300,
             tol = 1e-4,
             random_state = seed
             )

#Prediction 1, nclusters = 2, scaled dataset

scaled_predicted = km1.fit_predict(scaled_df)

print(scaled_predicted)

#Prediction 2, nclusters = 2, pca dataset

pca_predicted = km1.fit_predict(pca_df) # fit_predict = Compute cluster centers and predict cluster index for each sample.

print(pca_predicted)


#Check if they make the same prediction
print(sum(scaled_predicted - pca_predicted)) #yes, except for 3 observations

#add cluster column to both dfs

scaled_df['cluster'] = scaled_predicted
print(scaled_df)

pca_df['cluster'] = pca_predicted
print(scaled_df)




###estimate optimal number of clusters

## elbow method via SSE
#scaled
sse = []
for i in range(1, 11):
    km = KMeans(
        n_clusters=i, init='random',
        n_init=10, max_iter=300,
        tol=1e-04, random_state=0
    )
    km.fit(scaled_df)
    sse.append(km.inertia_)


#plot it

plt.plot(range(1, 11), sse, marker='o', color ='#FA8072')
plt.xlabel('Number of clusters')
plt.ylabel('SSE')
plt.show()


#PCA
sse = []
for i in range(1, 11):
    km = KMeans(
        n_clusters=i, init='random',
        n_init=10, max_iter=300,
        tol=1e-04, random_state=0
    )
    km.fit(pca_df)
    sse.append(km.inertia_)
    
#plot it

plt.plot(range(1, 11), sse, marker='o', color='#9ACD32')
plt.xlabel('Number of clusters')
plt.ylabel('SSE')
plt.show()   

##For both its roughly 3 clusters, but more leeway in the scaled df, therefore silhouette method as addition


##Silhouette method scaled df

score = silhouette_score(scaled_df, km.labels_, metric='euclidean')
print('Silhouette Score: %.3f' % score)


#Model 2

km2 = KMeans(n_clusters = 3,
             init = 'random',
             max_iter = 300,
             tol = 1e-4,
             random_state = seed
             )

#Prediction 2, nclusters = 3, scaled dataset

scaled_predicted2 = km2.fit_predict(scaled_df)

print(scaled_predicted2)

#Prediction 2, nclusters = 3, pca dataset

pca_predicted2 = km2.fit_predict(pca_df) # fit_predict = Compute cluster centers and predict cluster index for each sample.

print(pca_predicted2)

#add cluster column to both dfs

scaled_df['cluster2'] = scaled_predicted2
print(scaled_df)

pca_df['cluster2'] = pca_predicted2
print(scaled_df)

##Silhouette score scaled df

score = silhouette_score(scaled_df, km.labels_, metric='euclidean')
print('Silhouette Score: %.3f' % score)



# =============================================================================
# #Plot silhouettes
# 
# from yellowbrick.cluster import SilhouetteVisualizer
# 
# fig,ax = plt.subplots(2,2, figsize = (15,8))
# for i in [2,3,4,5]:
# 
#     # create kmeans instance for different numbers of clusters
#     km = KMeans(n_clusters=i, init= 'random', n_init =10, max_iter = 300, random_state = 0)
#     q, mod = divmod(i,2)
#     
#     #create visualiser
#     visualizer = SilhouetteVisualizer(km, colors = 'yellowbrick', ax=ax[q-1][mod])
#     visualizer.fit(df_standard)
# =============================================================================





## Analyze clusters k=2 scaled dataset

# load example dataset from seaborn 
sns.get_dataset_names()

# plot
sns.load_dataset('penguins')
sns.pairplot(scaled_df.iloc[:, 0:6], hue="cluster", corner = True) #iloc is to exclude the last column which is cluster2 from k = 3



## Analyze clusters k=2 PC dataset

# load example dataset from seaborn 
sns.get_dataset_names()

# plot
sns.load_dataset('penguins')
sns.pairplot(pca_df.iloc[:, 0:3], hue="cluster", corner = True) #iloc is to exclude the last column which is cluster2 from k = 3



## Analyze clusters k=3 scaled dataset

# load example dataset from seaborn 
sns.get_dataset_names()

# plot
sns.load_dataset('penguins')
sns.pairplot(scaled_df[['health', 'inflation', 'life_expec', 'gdpp', 'EXtoIM', 'cluster2']], hue="cluster2", corner = True)



## Analyze clusters k=3 PC dataset

# load example dataset from seaborn 
sns.get_dataset_names()

# plot
sns.load_dataset('penguins')
sns.pairplot(pca_df[['PC1', 'PC2', 'cluster2']], hue="cluster2", corner = True)



#Analyze clusters on original dataset

#Add clusters from scaled and PC dataset to OG dataset
scaled_df['cluster'] = scaled_predicted
print(scaled_df)

pca_df['cluster'] = pca_predicted
print(scaled_df)



##Export the PC df and scaled df with clusters

pca_df.to_csv(os.path.join(data_path,'pca_df_clustered.csv'),index=True)

scaled_df.to_csv(os.path.join(data_path,'scaled_df_clustered.csv'),index=True)