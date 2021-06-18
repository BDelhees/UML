# -*- coding: utf-8 -*-
"""
Created on Sat Jun  5 09:51:43 2021

@author: budde

UML 4: Results Analysis
"""

####Results Analysis
#Import basic libraries

import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


#Import data

data_path = "..."

#scaled df
scaled_df_final = pd.read_csv(f'{data_path}/scaled_df_final.csv', index_col = 0)

#make a scaled df onyl with colum of clusters

df_clusters = scaled_df_final[['Agglomerative_H']]


#Original df

original_df = pd.read_csv(f'{data_path}/Country-data.csv')

#set index for og df

original_df = original_df.set_index('country')



#merge datasets to get unscaled numbers

merged_df = df_clusters.merge(original_df, how = 'left', left_index=True, right_index=True)

#make ExtoIM ratio

merged_df['EXtoIM'] = merged_df['exports'] / merged_df['imports']

#drop the unused features

features_to_drop = ['income', 'child_mort', 'total_fer', 'imports', 'exports']
merged_df.drop(features_to_drop, axis=1, inplace=True)


###Mean and median for each cluster
#conditional mean

merged_df.groupby('Agglomerative_H').mean()


# =============================================================================
#                     health  inflation  life_expec         gdpp    EXtoIM
# Agglomerative_H                                                         
# 0                 4.792857  18.226786   69.503571  6367.071429  1.332500  Violet, emerging
# 1                 5.410000   7.719625   60.942500  1592.525000  0.660505  Red, developing
# 2                 6.585918   3.950184   74.795918  7423.469388  0.856947  Green, emerging
# 3                10.745000   7.330833   59.225000   971.250000  0.376489  Orange, developing
# =============================================================================

merged_df.groupby('Agglomerative_H').median()


# =============================================================================
#                  health  inflation  life_expec    gdpp    EXtoIM
# Agglomerative_H                                                 
# 0                 4.695      16.55       68.95  4450.0  1.240331  Violet, emerging 
# 1                 5.215       6.37       61.75   841.5  0.667557  Red, developing
# 2                 6.550       3.82       74.60  6250.0  0.861111  Green, emerging
# 3                11.200       6.14       59.20   579.0  0.395050  Orange, developing
# =============================================================================

#looking at individual clusters

mean_0_health = merged_df[(merged_df['Agglomerative_H'] == 0)]['health'].mean()

print(mean_0_health)

#count n in each cluster

merged_df[(merged_df['Agglomerative_H'] == 0)].count() #0 = 28

merged_df[(merged_df['Agglomerative_H'] == 1)].count() #1 = 40

merged_df[(merged_df['Agglomerative_H'] == 2)].count() #2 = 49

merged_df[(merged_df['Agglomerative_H'] == 3)].count() #3 = 12


#delete emerging nations clusters

merged_df = merged_df[merged_df.Agglomerative_H != 0]

merged_df = merged_df[merged_df.Agglomerative_H != 2]




#Make pairwise plots

sns.load_dataset('penguins')
sns.pairplot(merged_df.iloc[:,:], hue="Agglomerative_H", corner = True)

###Freedom House
#import freedom house dataset


fh_df = pd.read_csv(f'{data_path}/freedomhouse_data.csv', index_col = 0, sep = ';')

#only keep the total column

fh = fh_df[['Total']]

#rename countries that they match
pd.set_option('display.max_rows', 1000)
print(fh)

#fh has no data for gambia

fh = fh.rename(index={'Congo (Kinshasa)': 'Congo, Dem. Rep.', 'The Gambia': 'Gambia', 'Micronesia': 'Micronesia, Fed. Sts.', 'Laos': 'Lao'})


#merge 

fh_merged_df = fh.merge(merged_df, how = 'left', left_index=True, right_index=True)

fh_merged_df = fh_merged_df.dropna()


#Print to latex

fh_latex = fh_merged_df[['Total', 'Agglomerative_H']]

fh_latex = fh_latex.sort_values(by='Total', ascending=False)

print(fh_latex.to_latex())



#Export fh_latex

fh_latex.to_csv(os.path.join(data_path,'fh_latex.csv'),index=True)





