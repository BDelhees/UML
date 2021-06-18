# -*- coding: utf-8 -*-
"""
Created on Sat Jun 12 08:53:17 2021

@author: budde

UML 6: Results Analysis
"""

####World Map 
#Import basic libraries

import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


#Import data

data_path = "..."

#fh df
fh1 = pd.read_csv('fh_latex.csv', index_col = 0)


#make country column from index

fh1['Country_Region'] = fh1.index


# generate country code  based on country name 
import pycountry 
def alpha3code(column):
    CODE=[]
    for country in column:
        try:
            code=pycountry.countries.get(name=country).alpha_3
           # .alpha_3 means 3-letter country code 
           # .alpha_2 means 2-letter country code
            CODE.append(code)
        except:
            CODE.append('None')
    return CODE
# create a column for code 
fh1['CODE']=alpha3code(fh1.Country_Region)
fh1.head()

#Change country code of following countries:
# Source: https://en.wikipedia.org/wiki/List_of_ISO_3166_country_codes
    # Micronesia, Fed. Sts. = FSM
    # Moldova = MDA
    # Cote d'Ivoire = CIV 
    # Tanzania = TZA
    # Congo, Dem. Rep. = COD
    # Lao = LAO
    
fh1.loc[fh1.Country_Region == "Micronesia, Fed. Sts.", "CODE"] = "FSM"
fh1.loc[fh1.Country_Region == "Moldova", "CODE"] = "MDA"
fh1.loc[fh1.Country_Region == "Cote d'Ivoire", "CODE"] = "CIV"   
fh1.loc[fh1.Country_Region == "Tanzania", "CODE"] = "TZA"
fh1.loc[fh1.Country_Region == "Congo, Dem. Rep.", "CODE"] = "COD"
fh1.loc[fh1.Country_Region == "Lao", "CODE"] = "LAO"


# first let us merge geopandas data with our data
# 'naturalearth_lowres' is geopandas datasets so we can use it directly
import geopandas

world = geopandas.read_file(geopandas.datasets.get_path('naturalearth_lowres'))
# rename the columns so that we can merge with our data
world.columns=['pop_est', 'continent', 'name', 'CODE', 'gdp_md_est', 'geometry']
# then merge with our data 
merge=pd.merge(world,fh1,on='CODE')

print(merge.sort_values(by='name', ascending=False))
#missing countries
#tonga
#micronesia
#kiribati
#comoros

# last thing we need to do is - merge again with our location data which contains each country’s latitude and longitude
location=pd.read_csv('https://raw.githubusercontent.com/melanieshi0120/COVID-19_global_time_series_panel_data/master/data/countries_latitude_longitude.csv')
merge=merge.merge(location,on='name').sort_values(by='Total',ascending=False).reset_index()

print(fh1.sort_values(by='Country_Region', ascending=False))
#missing countries
#vanuatu
#lesotho
#cote d'ivoire

import mapclassify



# plot confirmed cases world map 
merge.plot(column='Total', scheme="quantiles",
           figsize=(40, 60),
           legend=True,cmap='RdYlGn')
plt.title('',fontsize=50)
# add countries names and numbers 
for i in range(0,10):
    plt.text(float(merge.longitude[i]),float(merge.latitude[i]),"{}\n{}".format(merge.name[i],merge.Total[i]),size=10)
plt.show()




