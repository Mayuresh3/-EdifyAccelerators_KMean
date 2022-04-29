import streamlit as st
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn import metrics
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as qo
from sklearn import datasets, neighbors
from mlxtend.plotting import plot_decision_regions
import json
from urllib.request import urlopen
import seaborn
from sklearn.decomposition import PCA


st.header("KMean")
st.title("K-Means Clustering")

data_file=st.file_uploader("Upload Dataset",type=["csv","excel"])
if data_file is not None:
    st.write(type(data_file))
    df=pd.read_csv(data_file)
    if data_file is not None:
        df_display = st.checkbox("Display Raw Data", value=False)

    if df_display:
        st.write(df)
    numerics = ['int16', 'int32', 'int64', 'float32', 'float64']
    df = df.select_dtypes(include=numerics)
    option1 = st.selectbox(
        'Select parameter 1: ',df.columns)

    st.write('You selected:', option1)
    i_1 = df.columns.get_loc(option1)

    option2 = st.selectbox(
        'Select parameter 2: ',df.columns)

    st.write('You selected:', option2)
    i_2 = df.columns.get_loc(option2)

        
    # -----------------------------------------------------------

    # Helper functions
    # -----------------------------------------------------------
    # Load data from external source
    #df = pd.read_csv(
    #  "marketing_segmentation.csv"
    #)
    # -----------------------------------------------------------

    # Sidebar
    # -----------------------------------------------------------

    # -----------------------------------------------------------


    # Main
    # -----------------------------------------------------------
    # Create a title for your app


    # A description
    #st.write("Here is the dataset used in this analysis:")

    # Display the dataframe
    #st.write(df)
    # -----------------------------------------------------------
    # Display the dataframe

    # SIDEBAR
    # -----------------------------------------------------------
sidebar = st.sidebar
    #df_display = sidebar.checkbox("Display Raw Data", value=True)

n_clusters = sidebar.slider(
    "Select Number of Clusters (k value) :",
    min_value=1,
    max_value=10,
)
    # -----------------------------------------------------------

    # Imports
    # -----------------------------------------------------------
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme()
    # -----------------------------------------------------------

    # Helper functions
    # -----------------------------------------------------------

def run_kmeans(df, n_clusters=1):
    kmeans = KMeans(n_clusters, random_state=0).fit(df[[option1,option2]])

    fig, ax = plt.subplots(figsize=(16, 9))

        #Create scatterplot
    ax = sns.scatterplot(
        ax=ax,
        x=df[option1],
        y=df[option2],
        hue=kmeans.labels_,
        palette=sns.color_palette("colorblind", n_colors=n_clusters),
        legend=None,
    )

    return fig
    # -----------------------------------------------------------

    # MAIN APP
    # -----------------------------------------------------------

    # Show cluster scatter plot
if st.checkbox('K Means graph'):
    if data_file is not None: 
        st.write(run_kmeans(df, n_clusters=n_clusters))


    #sc = StandardScaler()
    #df_scaled = sc.fit_transform(df) 
def Elbow(df_scaled):
    """ssq =[] 
    x_ax=[]
    for K in range(1,11):
        model = KMeans(n_clusters=K, random_state=123) 
        result = model.fit(df_scaled)
        ssq.append(model.inertia_)
        x_ax.append(K)
    d=pd.DataFrame({'x':x_ax,'y':ssq})
    """
    X= df[[option1,option2]]
    wcss=[]
    for i in range(1,15):
        kmeans=KMeans(n_clusters=i,init='k-means++',random_state=42)
        kmeans.fit(X)
        wcss.append(kmeans.inertia_)
    plt.plot()
    plt.title('The Elbow Method')
    plt.xlabel('Number of clusters')
    plt.ylabel('WCSS')
    plt.show()
    #st.line_chart(d.rename(columns={'x':'index'}).set_index('index'))
    fig = px.line(        
            df_scaled, #Data Frame
            x =range(1,15), #Columns from the data frame
            y = wcss,
            title = "Line frame"
        )
    fig.update_traces(line_color = "maroon")
    st.plotly_chart(fig)

         
if st.button('Elbow graph'):
 st.write(Elbow(df))
    # -----------------------------------------------------------