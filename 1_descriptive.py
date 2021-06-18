# -*- coding: utf-8 -*-
"""
Created on Sun May 30 08:36:49 2021

@author: budde


UML 1: Descriptive Analysis
"""

####Descriptive Analysis
#Import basic libraries

import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


#Import data

data_path = "..."

df = pd.read_csv(f'{data_path}/Country-data.csv')

#Check dataframe

df.head()
df.tail()

df.shape

#Check NAs and duplicates

df.isnull().sum() #No NAs

format(len(df[df.duplicated()])) #No duplicates


##Summary statistics for all variables

df_desc = round(df.describe().T, 2)

#Make Latex table
print(df_desc.to_latex())


#Coefficient of variation

cv = lambda x: np.std(x, ddof=1) / np.mean(x) * 100 

cv(df)

###Plots

##Histograms

#Child mortality

plt.figure(figsize=(12,5))
plt.title("Child Mortality")
ax = sns.histplot(df["child_mort"])
ax.set(xlabel=None)

#Exports

plt.figure(figsize=(12,5))
plt.title("Exports")
ax = sns.histplot(df["exports"])
ax.set(xlabel=None)

#Health

plt.figure(figsize=(12,5))
plt.title("Health")
ax = sns.histplot(df["health"])
ax.set(xlabel=None)

#Imports

plt.figure(figsize=(12,5))
plt.title("Imports")
ax = sns.histplot(df["imports"])
ax.set(xlabel=None)

#Income

plt.figure(figsize=(12,5))
plt.title("Income")
ax = sns.histplot(df["income"])
ax.set(xlabel=None)

#Inflation

plt.figure(figsize=(12,5))
plt.title("Inflation")
ax = sns.histplot(df["inflation"])
ax.set(xlabel=None)


#Life expectancy

plt.figure(figsize=(12,5))
plt.title("Life Expectancy")
ax = sns.histplot(df["life_expec"])
ax.set(xlabel=None)

#Fertility

plt.figure(figsize=(12,5))
plt.title("Total Fertility")
ax = sns.histplot(df["total_fer"])
ax.set(xlabel=None)


#GDP p.c.

plt.figure(figsize=(12,5))
plt.title("GDP p.c.")
ax = sns.histplot(df["gdpp"])
ax.set(xlabel=None)


##Correlation

df_corr = df.corr(method="pearson")
mask = np.zeros_like(df_corr)
mask[np.triu_indices_from(mask)] = True
plt.figure(figsize=(15,10))
sns.heatmap(df_corr, mask=mask, square=True, cmap='coolwarm', annot=True)


## Pairwise

g = sns.pairplot(df)
g.fig.set_size_inches(12,10)


### Pre-proccesing, Feature Engineering


#Create new variable, Export to Import Ratio

df['EXtoIM'] = df['exports'] / df['imports']

#Drop Income, child mortality and total fertility
features_to_drop = ['income', 'child_mort', 'total_fer', 'imports', 'exports']
df.drop(features_to_drop, axis=1, inplace=True)


#make hisogram for import to exports ratio

plt.figure(figsize=(12,5))
plt.title("Export to Import Ratio")
ax = sns.histplot(df["EXtoIM"])
ax.set(xlabel=None)

#standardize the dataset

df = df.set_index('country') #Set the country column as index

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaled_df = scaler.fit_transform(df)
scaled_data = pd.DataFrame(scaled_df, columns=df.columns, index=df.index) #Use the variable names and idnex from df for scaled dataframe 


##Summary statistics for scaled variables

scaled_desc = round(scaled_data.describe().T, 2)

#Make Latex table
print(scaled_desc.to_latex())


###Dimensionality Reduction

##PCA

from sklearn.decomposition import PCA

#fit and transform
pca = PCA()
pca.fit(scaled_data)
pca_scaled_data = pca.transform(scaled_data)

#formula for percentage variation calculation
per_var = np.round(pca.explained_variance_ratio_*100, decimals =1)
labels = ['PC' + str(x) for x in range (1, len(per_var)+1)]


# plot the percentage of explained variance by principal component
plt.bar(x=range(1,len(per_var)+1), height=per_var, tick_label = labels, color = '#9ACD32')
plt.ylabel('Percentage of Explained Variance')
plt.xlabel('Principal Component')
plt.title('Scree Plot')
plt.show() #the sum is always 100 of all PCAs combined

# plot pca
pca_df_standard = pd.DataFrame(pca_scaled_data, columns = labels)
plt.scatter(pca_df_standard.PC1, pca_df_standard.PC2, color = '#9ACD32')
plt.title('PCA')
plt.xlabel('PC1 - {0}%'.format(per_var[0]))
plt.ylabel('PC2 - {0}%'.format(per_var[1]))


#

#Use index from normal df for the PCA
pca_df_standard.index = df.index




##Export the PC df and scaled df

pca_df_standard.to_csv(os.path.join(data_path,'pca_df.csv'),index=True)

scaled_data.to_csv(os.path.join(data_path,'scaled_df.csv'),index=True)


