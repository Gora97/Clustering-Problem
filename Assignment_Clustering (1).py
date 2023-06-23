#!/usr/bin/env python
# coding: utf-8

# In[20]:


import pandas as pd 
import numpy as np
from sklearn.preprocessing import StandardScaler
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans,DBSCAN
import scipy.cluster.hierarchy as sch
from sklearn.cluster import AgglomerativeClustering


# In[2]:


data=pd.read_csv(r'crime_data.csv')


# In[3]:


data.head()


# In[4]:


data1=data.rename({'Unnamed: 0':'City'},axis=1)


# In[5]:


data1.head()


# In[7]:


data1.describe()


# In[9]:


data1.isnull().sum()


# # Normalization function 

# In[37]:


def norm_func(i):
    x = (i-i.min())/(i.max()-i.min())
    return (x)


# # Hierarchical Clustering 

# In[39]:


data1_norm = norm_func(data1.iloc[:,1:])


# In[90]:


data1_norm.head()


# In[76]:


# create dendrogram
dendrogram = sch.dendrogram(sch.linkage(data1_norm, method='average'))


# In[97]:


# create clusters
hc = AgglomerativeClustering(n_clusters=4, affinity = 'euclidean', linkage = 'average')


# In[98]:


# save clusters for chart
y_hc = hc.fit_predict(data1_norm)
Clusters=pd.DataFrame(y_hc,columns=['Clusters'])


# In[99]:


Clusters


# In[100]:


data1['clusterid_new'] = Clusters


# In[101]:


data1.sort_values("clusterid_new")


# # KMEANS

# In[92]:


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaled_data1_df = scaler.fit_transform(data1.iloc[:,1:])


# In[93]:


scaled_data1_df


# In[94]:


wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i,random_state=0)
    kmeans.fit(scaled_data1_df)
    wcss.append(kmeans.inertia_)
    
plt.plot(range(1, 11), wcss)
plt.title('Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()


# In[95]:


#Build Cluster algorithm
from sklearn.cluster import KMeans
clusters_new = KMeans(4, random_state=42)
clusters_new.fit(scaled_data1_df)


# In[96]:


clusters_new.labels_


# In[102]:


#these are standardized values.
clusters_new.cluster_centers_


# In[103]:


data1.groupby('clusterid_new').agg(['mean']).reset_index()


# In[105]:


data1.sort_values("clusterid_new")


# # DBSCAN

# In[126]:


dbscan = DBSCAN(eps=1.0, min_samples=5)

dbscan.fit(scaled_data1_df)


# In[127]:


dbscan.labels_


# In[128]:


cl=pd.DataFrame(dbscan.labels_,columns=['cluster'])


# In[129]:


cl


# In[122]:


data1['clusterid_new'] = cl


# In[123]:


data1.sort_values("clusterid_new")


# In[ ]:




