# Insurance Customer Segmentation Project

## By Alex Ordonez and Vishaka Sharma

This project implements customer segmentation for an insurance company using various clustering methods. The goal is to identify meaningful customer segments to assist strategic decision-making and targeted marketing efforts.

Project Structure:

1. Data

    - synthetic_insurance_data.csv: A synthetic dataset containing demographic and policy-related information of insurance clients used for clustering analysis.

2. Analysis and Reports

    - InsuranceCustomerSegmentation_FinalReport.docx: Detailed report covering methodology, analysis, results, and interpretations of the clustering models.

    - Insurance_Customers_GroupProject.ipynb: Jupyter Notebook containing:

        * Dataset Overview

        * Data Cleaning & Preprocessing

        * Exploratory Data Analysis (EDA)

        * Implementation and evaluation of clustering methods:

        * K-Means Clustering

        * Gaussian Mixture Models (GMM)

        * Hierarchical Clustering

        * DBSCAN

        * Visualization of clustering results

3. Streamlit Application

    - Main.py: Interactive Streamlit web application enabling users to:

        * Upload their own datasets

        * Conduct Exploratory Data Analysis (EDA) on selected columns

        * Perform clustering using various algorithms provided in the application

        * Clustering Methods Used

        * K-Means Clustering: Partition-based clustering algorithm.

        * Gaussian Mixture Models (GMM): Probabilistic clustering approach based on Gaussian distributions.

        * DBSCAN: Density-based clustering method ideal for finding arbitrarily shaped clusters.

        * Hierarchical Clustering: Agglomerative approach creating a dendrogram-based hierarchy of clusters.

## Dependencies

- Ensure you have the following Python libraries installed:

    * pip install streamlit pandas numpy scikit-learn scipy kneed matplotlib seaborn

    * streamlit: For web application deployment

    * pandas, numpy: Data manipulation and numerical operations

    * sklearn, scipy: Clustering algorithms and scientific computing

    * kneed: Optimal cluster identification

    * matplotlib, seaborn: Data visualization

- How to Run the Streamlit App

    > streamlit run Main.py

This command will launch the web application in your default browser, allowing you to upload data and interactively perform EDA and clustering analysis.

