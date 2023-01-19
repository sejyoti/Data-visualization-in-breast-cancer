#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt


# In[9]:


#Matplotlib is a popular vislization Library,but it has flows
#Flaws are defaults not ideal
#Library low level
#Lack of integration with pandas data structure


#MATPLOT WRAPPER
#BOXPLOT IS A MORE COMPLICATED BUT HELPFUL VISUALIZATION IN MATPLOTLIB,PANDAS,WRAPPERS,SEABORN
#Bloxplot is a stdardized way of displaying the distribution of data baed on five number summary
#BLOXPLOT
#Tell you the value of your outliers
#Identify if data is symmetrical
#Determine how tightly data is grouped
#See if your data is skewed
#The goal of the visualization is to show how the distributions for the column area_mean differs for benign vs malignant diagnosis



#Load Winconsis breast cancer dataset
#either benign or malignant
filename= ('wisconsinBreastCancer.csv')
cancer_df = pd.read_csv('wisconsinBreastCancer.csv')


# In[11]:


cancer_df.head()


# In[13]:


cancer_df ['diagnosis'].value_counts (dropna=False)


# In[14]:


#PLOTTING USING MATPLOTLIB

malignant = cancer_df.loc[cancer_df['diagnosis']=='M','area_mean'].values
benign = cancer_df.loc[cancer_df['diagnosis']=='B','area_mean'].values

plt.boxplot([malignant,benign],labels =['M','B']);


# In[15]:


#plotting using pandas
#Pandas can be as a wrapper around Matplotlib.One reason why you might want to plot using pandas is that it requies less code
#we are going to create a boxplot
#to show how much less syntax you need to create the plot with pandas vs matplotlib

#Getting rid of area_mean

cancer_df.boxplot(column = 'area_mean' , by ='diagnosis');


# In[16]:


cancer_df.boxplot(column = 'area_mean' , by ='diagnosis');
plt.title('');
plt.suptitle('');


# In[23]:


#plotting seaborn
#Wrappers around the functions are also knows as decorators which are a very powerful and useful tool in Python 
#since it allows programmers to modify the behavior of function or class. 
#Decorators allow us to wrap another function in order to extend the behavior of the wrapped function, 
#without permanently modifying it

#• Close integration with pandas data structures
#• Dataset oriented API for examining relationships between multiple variables.
#Specialized support for using categorical variables to show observations or aggregate statistics.
#Concise control over matplotlib figure styling with several built-in themes.
#• Tools for choosing color palettes that faithfully reveal patterns in your data.

import seaborn as sns

sns.boxplot(x='diagnosis', y='area_mean' ,data=cancer_df)


# In[2]:


#Heatmap
#Heatmap is a graphical representation of data where values are depicted by colors
#Sequentical (low values to high values)
#Qualitative(discrete chynks of data)

#The inline flag will use the appropriate backend to make figures appear inline in the not
get_ipython().run_line_magic('matplotlib', 'inline')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[4]:


#LOAD DATA

#The data is a confusion matrix ehich is a table that is often used describe the performance of the machine learning clasiification model
#it tells you where the prediction went wrong
filename ='digitsDataset.csv'
df = pd.read_csv('digitsDataset.csv')


# In[6]:


df


# In[8]:


#Histogram gives you a general idea how your data looks like

filename = ('kingCountyHouseData.csv')
df = pd.read_csv('kingCountyHouseData.csv')


# In[9]:


#A summary of the variation in a measued variable
#its a frequency distribution
#Histograms work by binning the entire range of values into a series of intervals, 
#and then counting how many values fall into each interval.
df.head()


# In[11]:


df['price'].head()


# In[12]:


df['price'].hist()


# In[13]:


df['price'].hist()
plt.xticks(rotation=90)


# In[14]:


plt.style.use('seaborn')


# In[15]:


df['price'].hist(bins=30)


# In[16]:


#Visualising a subset of data]
price_filter = df.loc[:,'price']<=3000000
df.loc[price_filter,'price'].hist(bins=30)


# In[17]:


price_filter = df.loc[:,'price']<=3000000
df.loc[price_filter,'price'].hist(bins=30,
                                  edgecolor='black')


# In[18]:


#Subplot
#It is often useful to compare different subset of data side by side
filename ='digitsDataset.csv'
df = pd.read_csv('digitsDataset.csv')


# In[19]:


df.head()


# In[20]:


#Show image
pixel_colnames = df.columns[:-1]


# In[21]:


pixel_colnames


# In[ ]:




