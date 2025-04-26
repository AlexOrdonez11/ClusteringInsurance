import streamlit as st
import pandas as pd
import numpy as np
from math import ceil
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN
from scipy.cluster.hierarchy import linkage, fcluster
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score
from kneed import KneeLocator
import matplotlib.pyplot as plt
import seaborn as sns


    
def dbscan(df_scaled):

    # Fit pca to do some reduction for DBScan
    pca = PCA().fit(df_scaled)

    # Calculate cumulative explained variance
    cumulative_variance = np.cumsum(pca.explained_variance_ratio_)

    # Find the number of components that explain at least 80% of the variance
    n_components_80 = np.argmax(cumulative_variance >= 0.80) + 1

    # Get pca components
    pca = PCA(n_components=n_components_80)

    # Fit the data
    X_reduced = pca.fit_transform(df_scaled)

    # Calculate distances
    neighbors = NearestNeighbors(n_neighbors=n_components_80)
    neighbors_fit = neighbors.fit(X_reduced)
    distances, indices = neighbors_fit.kneighbors(X_reduced)

    # Calculate elbow to determine best epsilon number
    k_distances = np.sort(distances[:, n_components_80-1])
    knee = KneeLocator(range(len(k_distances)), k_distances, curve='convex', direction='increasing')
    best_eps = k_distances[knee.knee] if knee.knee is not None else 0.5

    # Fit DBScan
    dbscan = DBSCAN(eps=best_eps, min_samples=n_components_80)
    return dbscan.fit_predict(df_scaled)

def kMeans(df_scaled):

    silhouette_scores = []
    K_range = range(2, 10)

    # Getting best K fitting the model multiple times until we get the best silhouette score
    for k in K_range:
        kmeans = KMeans(n_clusters=k, random_state=42)
        labels = kmeans.fit_predict(df_scaled)
        silhouette_scores.append(silhouette_score(df_scaled, labels))

    # Save best k 
    optimal_k= K_range[np.argmax(silhouette_scores)]

    # Fit the Model using the best k 
    kmeans = KMeans(n_clusters=optimal_k, random_state=42)
    return kmeans.fit_predict(df_scaled)

def Hier(df_scaled):
    Z = linkage(df_scaled, method='ward')

    silhouette_scores = []
    k_range = range(2, 11)

    # Getting best K fitting the model multiple times until we get the best silhouette score
    for k in k_range:
        labels = fcluster(Z, k, criterion='maxclust')
        score = silhouette_score(df_scaled, labels)
        silhouette_scores.append(score)

    # Save best k 
    optimal_k = k_range[np.argmax(silhouette_scores)]

    return fcluster(Z, optimal_k, criterion='maxclust')

def GMM(df_scaled):
    k_range = range(1, 10)
    silhouette_scores=[]

    # Getting best K fitting the model multiple times until we get the best silhouette score
    for k in k_range:
        gmm = GaussianMixture(n_components=k, random_state=42)
        gmm.fit(df_scaled)
        labels=gmm.fit_predict(df_scaled)
        if len(np.unique(labels)) > 1:
            score = silhouette_score(df_scaled, labels)
            silhouette_scores.append(score)
        else:
            silhouette_scores.append(-1)

    # Save best k 
    optimal_k = k_range[np.argmax(silhouette_scores)]

    # Fit the Model using the best k 
    gmm=GaussianMixture(n_components=optimal_k, random_state=42)
    return gmm.fit_predict(df_scaled)


def safe_silhouette(data, labels, name):

    # Calculate Silhouette Scores and print the results
    if len(set(labels)) > 1 and len(set(labels)) < len(labels):
        score = silhouette_score(data, labels)
        st.write(f"Silhouette Score - {name}: {score:.3f}")
        return score
    else:
        st.write(f"Silhouette Score - {name}: Not applicable (only one cluster or too noisy)")
        return None

if __name__== "__main__":
    st.write("Select a file to begging with the app")
    # Using st.file_uploader to upload a CSV file
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

    if uploaded_file is not None:
        # Reading the CSV file into a Pandas DataFrame
        dfp = pd.read_csv(uploaded_file)

        # Assign ID
        dfp['id'] = dfp.index
        df=dfp

        # Displaying the DataFrame
        st.write("DataFrame:")
        st.write(df.describe())

        
        datachecks=[]
        iteration=0

        st.write("Select the columns(features) to do the clustering:")
        # Check boxes based on features, on 2 columns
        col1,col2 = st.columns(2)
        for i in df.columns:
            if not i == 'id':
                if iteration % 2==0:
                    with col1: 
                        datachecks.append(st.checkbox(i,value=True))
                    iteration+=1

                else:
                    with col2: 
                        datachecks.append(st.checkbox(i,value=True))
                    iteration+=1

        if st.button("Generate"):
            st.session_state.generate = True

        # if the button was clicked
        if st.session_state.get("generate"):
            datacolumns=[]
            count=0
            # Only keep the columns selected
            for i in df.columns:
                if not i == 'id':
                    if datachecks[count]:
                        datacolumns.append(i)
                    count+=1
            df = df[datacolumns] 
            st.write("Count Nulls:")
            st.write(df.isna().sum())

            # Getting number of NA
            count=df.isna().sum().sum();
            
            # FIll N/A with median
            df.fillna(df.median(numeric_only=True), inplace=True)

            # Identify numerical columns for correlation matrix
            numeric_cols = df.select_dtypes(include=[np.number])
            
            # Show Correlation Matrix
            st.write("Correlation Matrix")
            corr_matrix = numeric_cols.corr()
            corrplot = plt.figure(figsize=(18, 15))
            sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", linewidths=0.5)
            plt.title("Feature Correlation Matrix")
            st.pyplot(corrplot)
            plt.close(corrplot)

            # Let user build plots
            st.write("If you want to generate some EDA plots click on Generate Plot, if not continue")
            if st.button("Generate Plot"):
                st.session_state.step = "plot"
            
            # If he wants to build plots
            if st.session_state.get("step")=="plot":

                st.write("Select 1 numerical column for histogram")
                st.write("Select 2 numerical columns for scatterplot")
                st.write("Select 1 numerical & 1 categorical column for boxplots")
                datachecks2=[]
                iteration2=0

                # Another list of chenk boxes for ploting
                col3,col4 = st.columns(2)
                for i in df.columns:
                    if iteration2 % 2==0:
                        with col3: 
                            datachecks2.append(st.checkbox("Plot "+i))
                        iteration2+=1

                    else:
                        with col4: 
                            datachecks2.append(st.checkbox("Plot "+i))
                        iteration2+=1
                if st.button("Plot"):
                    datacolumns2=[]
                    count2=0
                    # Get Columns Selected
                    for i in df.columns:
                        if datachecks2[count2]:
                            datacolumns2.append(i)
                            print(i)
                        count2+=1

                    # If the user selected 1 numerical column show a histogram, 1 numerical and 1 categorical boxplot distribution and 
                    # 2 numerical a scatter plot.
    
                    if len(datacolumns2)==2:
                        # 2 numerical
                        if df[datacolumns2[0]].dtype in ['int64', 'float64'] and df[datacolumns2[1]].dtype in ['int64', 'float64']:
                            plot=plt.figure(figsize=(15,10))
                            sns.scatterplot(x=df[datacolumns2[0]], y=df[datacolumns2[1]],
                            palette='tab10')
                            plt.title(datacolumns2[0] + ' by ' + datacolumns2[1])
                            plt.xlabel(datacolumns2[0])
                            plt.ylabel(datacolumns2[1])
                            st.pyplot(plot.figure)
                            plt.close(plot.figure)

                        # 1 numerical and 1 categorical
                        elif df[datacolumns2[0]].dtype in ['int64', 'float64']:
                            plot=plt.figure(figsize=(10,6))
                            sns.boxplot(x=datacolumns2[1], y=datacolumns2[0], data=df, palette='Set2')
                            plt.title(datacolumns2[0] + 'by' + datacolumns2[1])
                            plt.xlabel(datacolumns2[1])
                            plt.ylabel(datacolumns2[0])
                            st.pyplot(plot.figure)
                            plt.close(plot.figure)

                        # 1 categorical and 1 numerical
                        elif df[datacolumns2[1]].dtype in ['int64', 'float64']:
                            plot=plt.figure(figsize=(10,6))
                            sns.boxplot(x=datacolumns2[0], y=datacolumns2[1], data=df, palette='Set2')
                            plt.title(datacolumns2[1] + 'by' + datacolumns2[0])
                            plt.xlabel(datacolumns2[0])
                            plt.ylabel(datacolumns2[1])
                            st.pyplot(plot.figure)
                            plt.close(plot.figure)
                        else:
                            st.write("You must select either 1 or 2 columns (At least 1 must be numerical)")

                    elif len(datacolumns2)==1:
                        # 1 numerical
                        if df[datacolumns2[0]].dtype in ['int64', 'float64']:
                            plot=plt.figure(figsize=(10,6))
                            sns.histplot(df[datacolumns2[0]], bins=20, kde=True, color='skyblue')
                            plt.title(datacolumns2[0]+" "+'Distribution')
                            plt.xlabel(datacolumns2[0])
                            plt.ylabel('Count')
                            st.pyplot(plot.figure)
                            plt.close(plot.figure)
                        else:
                            st.write("You must select either 1 or 2 columns (At least 1 must be numerical)")
                    else:
                        st.write("You must select either 1 or 2 columns (At least 1 must be numerical)")

            # Give chance to remove Outliers with IQR
            st.write("Decide if you want to remove or keep outliers:")
            outliers=st.checkbox("Check to Remove Outliers",value=True)
            
            if st.button("Continue with Clustering"):
                st.session_state.step = "cluster"

            if st.session_state.get("step")=="cluster":
                
                # Get categorical variables
                categorical_cols = df.select_dtypes(include=['object']).columns

                # Detecting more categorical columns
                for i in df.drop(categorical_cols, axis=1).columns.values:
                    if df[i].nunique()<11:
                        categorical_cols=np.append(categorical_cols, i)

                st.write("Categorical Columns:", categorical_cols)

                # One-Hot Encoding
                df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

                # Identify numerical columns
                numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns

                if outliers:
                    #Compute IQR only for numeric columns
                    Q1 = df[numeric_cols].quantile(0.25)
                    Q3 = df[numeric_cols].quantile(0.75)
                    IQR = Q3 - Q1

                    # Remove extreme outliers
                    dfp= dfp[~((df[numeric_cols] < (Q1 - 1.5 * IQR)) | (df[numeric_cols] > (Q3 + 1.5 * IQR))).any(axis=1)]
                    df = df[~((df[numeric_cols] < (Q1 - 1.5 * IQR)) | (df[numeric_cols] > (Q3 + 1.5 * IQR))).any(axis=1)]

                    
                    # Plot boxplots of numerical columns *after* outlier removal
                    st.write("Distribution of Numerical Features after outlier removal")
                    fig, axes = plt.subplots(2, ceil(len(numeric_cols) / 2), figsize=(20, 11))
                    fig.subplots_adjust(hspace=0.5)

                    for ax, feat in zip(axes.flatten(), numeric_cols):
                        sns.boxplot(x=df[feat], ax=ax)
                        ax.set_title(f"{feat}")

                    # Show distribution after outlier removal
                    plt.suptitle("Metric Feature Distributions After Outlier Removal", fontsize=20)
                    plt.tight_layout(rect=[0, 0, 1, 0.95])
                    st.pyplot(fig)
                    plt.close(fig)
                    st.write("Outliers Removed based on IQR")

                # Standarize data
                X_scaled = StandardScaler().fit_transform(df)

                # Clustering Methods
                df['DBSCAN_Cluster']=dbscan(X_scaled)
                st.write("DBScan Clustering Done")
                df['KMeans_Cluster']=kMeans(X_scaled)
                st.write("KMeans Clustering Done")
                df['HierCluster']= Hier(X_scaled)
                st.write("Hierarchical Clustering Done")
                df['GMM_Cluster']= GMM(X_scaled)
                st.write("Gaussian Mixture Model Done")

                # Print Silhouette Scores
                score_kmeans = safe_silhouette(X_scaled, df['KMeans_Cluster'], 'KMeans')
                score_gmm = safe_silhouette(X_scaled, df['GMM_Cluster'], 'GMM')
                score_hier = safe_silhouette(X_scaled, df['HierCluster'], 'Hierarchical')
                score_dbscan = safe_silhouette(X_scaled, df['DBSCAN_Cluster'], 'DBSCAN')

                # Plot Clusters based on 2 reduced pca components
                pca = PCA(n_components=2)
                pca_data = pca.fit_transform(X_scaled)

                fig, axs = plt.subplots(2, 2, figsize=(18, 12))

                # KMeans
                sns.scatterplot(x=pca_data[:, 0], y=pca_data[:, 1], hue=df['KMeans_Cluster'],
                                    palette='tab10', ax=axs[0, 0])
                axs[0, 0].set_title("KMeans Clustering")

                # GMM
                sns.scatterplot(x=pca_data[:, 0], y=pca_data[:, 1], hue=df['GMM_Cluster'],
                                    palette='tab10', ax=axs[0, 1])
                axs[0, 1].set_title("GMM Clustering")

                # Hierarchical
                sns.scatterplot(x=pca_data[:, 0], y=pca_data[:, 1], hue=df['HierCluster'],
                                    palette='tab10', ax=axs[1, 0])
                axs[1, 0].set_title("Hierarchical Clustering")

                # DBSCAN
                sns.scatterplot(x=pca_data[:, 0], y=pca_data[:, 1], hue=df['DBSCAN_Cluster'],
                                    palette='tab10', ax=axs[1, 1])
                axs[1, 1].set_title("DBSCAN Clustering")

                # Set titles and labels
                for ax in axs.flat:
                    ax.set_xlabel("PC1")
                    ax.set_ylabel("PC2")
                    ax.legend(title="Cluster")

                plt.suptitle("PCA Visualization of Clustering Algorithms", fontsize=16)
                plt.tight_layout()
                    
                st.pyplot(fig)
                plt.close(fig)

                # Display Cluster count per model
                st.write("Cluster Counts:")
                st.write(df[['KMeans_Cluster', 'GMM_Cluster', 'HierCluster', 'DBSCAN_Cluster']].nunique())

                dfp['DBSCAN_Cluster'] = df['DBSCAN_Cluster']
                dfp['KMeans_Cluster'] = df['KMeans_Cluster']
                dfp['HierCluster'] = df['HierCluster']
                dfp['GMM_Cluster'] = df['GMM_Cluster']

                csv = dfp.to_csv(index=False).encode('utf-8')

                # Option to download the clusters
                st.download_button(
                    label="📥 Download CSV",
                    data=csv,
                    file_name='data.csv',
                    mime='text/csv'
                )

