import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans

df = pd.read_csv('starbucks-menu-nutrition-drinks.csv')

df['Calories'] = df['Calories'].str.replace('-','0')
df['Fat (g)'] = df['Fat (g)'].str.replace('-','0')
df['Carb. (g)'] = df['Carb. (g)'].str.replace('-','0')
df['Fiber (g)'] = df['Fiber (g)'].str.replace('-','0')
df['Protein'] = df['Protein'].str.replace('-','0')
df['Sodium'] = df['Sodium'].str.replace('-','0')

df['Calories'] = pd.to_numeric(df['Calories'])
df['Fat (g)'] = pd.to_numeric(df['Fat (g)'])
df['Carb. (g)'] = pd.to_numeric(df['Carb. (g)'])
df['Fiber (g)'] = pd.to_numeric(df['Fiber (g)'])
df['Protein'] = pd.to_numeric(df['Protein'])
df['Sodium'] = pd.to_numeric(df['Sodium'])

df.rename(index=str, columns={'Fat (g)' : 'fat','Carb. (g)' : 'carbo','Fiber (g)' : 'fiber'}, inplace=True)

x = df.drop(['Unnamed: 0','fat','fiber','Protein'], axis=1)

#show data
st.header("Isi Dataset")
if st.button('Tampilkan dataset'):
    st.write(x)

#elbow
clusters=[]
for i in range(1,11):
  kmeans = KMeans(n_clusters=i).fit(x)
  clusters.append(kmeans.inertia_)

fig,ax=plt.subplots(figsize=(12,8))
sns.lineplot(x=list(range(1,11)), y=clusters, ax=ax)
ax.set_title('mencari elbow')
ax.set_xlabel('clusters')
ax.set_xlabel('inertia')

st.header("Elbow Point")
st.set_option('deprecation.showPyplotGlobalUse', False)
elbo_plot=st.pyplot()

st.sidebar.header("Clustering Menu Minuman Starbucks")
st.sidebar.subheader("Nama : Salma Aulia")
st.sidebar.subheader("NIM : 211351134")
clust = st.sidebar.slider("Pilih Jumlah Cluster :", 2,10,3,1)
selectbox_x = st.sidebar.selectbox(
    "Input kolom X untuk visualisasi clustering",
    ('Calories', 'Carbo', 'Sodium')
)
if 'Calories' in selectbox_x: # If user selects Email  do 👇
    x_plot = x['Calories']
elif 'Carbo' in selectbox_x: # If user selects Email  do 👇
    x_plot = x['carbo']
elif 'Sodium' in selectbox_x: # If user selects Email  do 👇
    x_plot = x['Sodium']

selectbox_y = st.sidebar.selectbox(
    "Input kolom Y untuk visualisasi clustering",
    ('Calories', 'Carbo', 'Sodium')
)
if 'Calories' in selectbox_y: # If user selects Email  do 👇
    y_plot = x['Calories']
elif 'Carbo' in selectbox_y: # If user selects Email  do 👇
    y_plot = x['carbo']
elif 'Sodium' in selectbox_y: # If user selects Email  do 👇
    y_plot = x['Sodium']

def k_means(n_clust,x_plot,y_plot):
    kmean = KMeans(n_clusters=n_clust).fit(x)
    x['Labels'] = kmean.labels_

    pal = ["#682F2F","#B9C0C9", "#9F8A78","#F3AB60"]

    pl = sns.scatterplot(x=x_plot, y=y_plot,hue=x["Labels"], palette= pal)
    pl.set_title(f"Clustering berdasarkan {selectbox_x} dan {selectbox_y}")
    for label in x['Labels']:
        pl.annotate(label,
            (x[x['Labels']==label]['Calories'].mean(),
            x[x['Labels']==label]['Sodium'].mean()),
            horizontalalignment = 'center',
            verticalalignment = 'center',
            size = 20, weight='bold',
            color='black')
    st.header("Cluster Plot")
    st.pyplot()
    st.write(x)
    plt.legend()
    plt.show()

k_means(clust,x_plot,y_plot)