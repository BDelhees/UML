# -*- coding: utf-8 -*-
"""
Created on Tue Jun  1 08:38:57 2021

@author: budde


UML 3: Clustering Hierarchical
"""

####Hierarchical Clustering
#Import basic libraries

import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


#Import data

data_path = "..."

#PCA df
pca_df_c = pd.read_csv(f'{data_path}/pca_df_clustered.csv', index_col = 0)

#drop cluster2 column

pca_df_c = pca_df_c[['PC1', 'PC2', 'cluster']]

#drop developed nation, cluster = 0 

pca_df_c = pca_df_c[pca_df_c.cluster != 0]

#remove the k-menas cluster column, its no longer needed

pca_df_c = pca_df_c[['PC1', 'PC2']]






#Scaled df

scaled_df_c = pd.read_csv(f'{data_path}/scaled_df_clustered.csv', index_col = 0)

#drop cluster2 column

scaled_df_c = scaled_df_c[['health', 'inflation', 'life_expec', 'gdpp', 'EXtoIM', 'cluster']]

#drop developed nation, cluster = 0 

scaled_df_c = scaled_df_c[scaled_df_c.cluster != 0]

#remove the k-menas cluster column, its no longer needed

scaled_df_c = scaled_df_c[['health', 'inflation', 'life_expec', 'gdpp', 'EXtoIM']]



#Plot the new datasets without the developed nations

#PC
g = sns.pairplot(pca_df_c, corner = True)
g.fig.set_size_inches(12,10)

#Scaled

g = sns.pairplot(scaled_df_c, corner = True)
g.fig.set_size_inches(12,10)





### Hierarchical clustering

#libraries
from sklearn.cluster import KMeans, AgglomerativeClustering, AffinityPropagation
from sklearn.cluster import KMeans, AgglomerativeClustering, AffinityPropagation
from sklearn.cluster import MeanShift, estimate_bandwidth, SpectralClustering
from sklearn.cluster import DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score
import scipy.cluster.hierarchy as shc

#seed
seed = 45123


# Model 1, scaled

#Make dendogram to know the number of clusters

plt.figure(figsize=(20, 60))
_ = shc.dendrogram(shc.linkage(scaled_df_c, method='ward'), orientation = 'left', labels = scaled_df_c.index, leaf_font_size=16)

#Nr of clusters = 4
#Drop Nigeria, Outlier.

scaled_df_c = scaled_df_c[scaled_df_c.index != 'Nigeria']


#Make dendogram 2 without Nigeria

plt.figure(figsize=(30, 60))
_ = shc.dendrogram(shc.linkage(scaled_df_c, method='ward'), orientation = 'left', labels = scaled_df_c.index, leaf_font_size=16)


#make the model
hm1 = AgglomerativeClustering(n_clusters=4, affinity='euclidean', linkage='ward')
hm1.fit(scaled_df_c)
scaled_df_c['Agglomerative_H'] = hm1.labels_

#plot the m1

scaled_predicted = hm1.fit_predict(scaled_df_c)

print(scaled_predicted)

# load example dataset from seaborn 
sns.get_dataset_names()

# plot
sns.load_dataset('penguins')
sns.pairplot(scaled_df_c.iloc[:,:], hue="Agglomerative_H", corner = True)

#Description

#cluster 0 = Violet, emerging nations
#cluster 1 = Red, developing
#cluster 2 = green, emerging
#cluster 3 = orange, developing



#Model 2, PC

#Make dendogram to know the number of clusters

plt.figure(figsize=(30, 60))
_ = shc.dendrogram(shc.linkage(pca_df_c, method='ward'), orientation = 'left', labels = pca_df_c.index, leaf_font_size=16)

#Nr of clusters = 2



#make the model
hm2 = AgglomerativeClustering(n_clusters=2, affinity='euclidean', linkage='ward')
hm2.fit(pca_df_c)
pca_df_c['Agglomerative_H'] = hm2.labels_

#plot the m1

pca_predicted = hm2.fit_predict(pca_df_c)

print(pca_predicted)

# load example dataset from seaborn 
sns.get_dataset_names()

# plot
sns.load_dataset('penguins')
sns.pairplot(pca_df_c.iloc[:,:], hue="Agglomerative_H", corner = True)




###Create new df to see overlap between clusters

#make copy of scaled_df

copy_df = scaled_df_c.copy(deep=True)


#change cluster values to compare cluster overlap of scaled df to pc df
#emerging market clusters (0,2) and developing clusters (1,3) 

#make placeholders
copy_df['Agglomerative_H'] = copy_df['Agglomerative_H'].replace([2],5)
copy_df['Agglomerative_H'] = copy_df['Agglomerative_H'].replace([0],5)

copy_df['Agglomerative_H'] = copy_df['Agglomerative_H'].replace([3],6)
copy_df['Agglomerative_H'] = copy_df['Agglomerative_H'].replace([1],6)

#change them again to the values to compare to pc dataset

copy_df['Agglomerative_H'] = copy_df['Agglomerative_H'].replace([5],1)

copy_df['Agglomerative_H'] = copy_df['Agglomerative_H'].replace([6],0)


#rename column

copy_df = copy_df.rename(columns = {'Agglomerative_H':'scaled'}, inplace = False)

#drop all other columns

copy_df = copy_df.drop(columns=['health', 'inflation', 'gdpp', 'life_expec', 'EXtoIM'])


#make copy of pc_df

copy_df2 = pca_df_c.copy(deep=True)

#drop all other columns

copy_df2 = copy_df2.drop(columns=['PC1', 'PC2'])

#rename column

copy_df2 = copy_df2.rename(columns = {'Agglomerative_H':'pc'}, inplace = False)


#merge both 

merged_df = copy_df2.merge(copy_df, how = 'left', left_index=True, right_index=True)


merged_df = merged_df.dropna()
#compare the columns

from sklearn.metrics import jaccard_score

jaccard_score(merged_df['scaled'], merged_df['pc'])


#Scaled dataset is more granular, allows for better narrowing down, use this one

scaled_df_c.to_csv(os.path.join(data_path,'scaled_df_final.csv'),index=True)

