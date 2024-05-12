import pandas as pd
import numpy as np
import psycopg2
import sys
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def connect():

    conn = None
    try:
        print('Connecting..')
        conn = psycopg2.connect(database="xdrDb", user="admin", password="admin", host="localhost", port="5433")
        
    except (Exception, psycopg2.DatabaseError) as error:
            print("Error: %s" % error)
            sys.exit(1)   
 
        
    print("All good, Connection successful!")
    return conn

conn = connect()

telecom_data = pd.read_sql_query('SELECT * FROM xdr_data;', conn)  

telecom_data.fillna(telecom_data.mean(), inplace=True)

basic_metrics = telecom_data.describe()

dispersion_parameters = telecom_data.describe().loc[['std', 'min', '25%', '50%', '75%', 'max']]

plt.figure(figsize=(12, 6))
sns.histplot(data=telecom_data['Session_duration'], bins=30, kde=True)
plt.title('Distribution of Session Duration')
plt.xlabel('Session Duration')
plt.ylabel('Frequency')
plt.show()

bivariate_analysis = telecom_data[['Social_Media_DL', 'Google_DL', 'Email_DL', 'Youtube_DL', 
                                   'Netflix_DL', 'Gaming_DL', 'Other_DL', 'Total_DL']].corr()

telecom_data['Decile_Class'] = pd.qcut(telecom_data['Session_duration'], q=10, labels=False)
decile_data = telecom_data.groupby('Decile_Class')['Total_DL', 'Total_UL'].sum()

correlation_matrix = telecom_data[['Social_Media_DL', 'Google_DL', 'Email_DL', 'Youtube_DL', 
                                   'Netflix_DL', 'Gaming_DL', 'Other_DL']].corr()


scaler = StandardScaler()
scaled_data = scaler.fit_transform(telecom_data[['Social_Media_DL', 'Google_DL', 'Email_DL', 
                                                 'Youtube_DL', 'Netflix_DL', 'Gaming_DL', 'Other_DL']])

pca = PCA(n_components=2)
principal_components = pca.fit_transform(scaled_data)

principal_df = pd.DataFrame(data=principal_components, columns=['PC1', 'PC2'])

telecom_data_with_pca = pd.concat([telecom_data, principal_df], axis=1)


basic_metrics.to_csv('basic_metrics.csv')
dispersion_parameters.to_csv('dispersion_parameters.csv')
bivariate_analysis.to_csv('bivariate_analysis.csv')
decile_data.to_csv('decile_data.csv')
correlation_matrix.to_csv('correlation_matrix.csv')
telecom_data_with_pca.to_csv('telecom_data_with_pca.csv')


import pandas as pd
from sklearn.metrics import euclidean_distances
from sklearn.cluster import KMeans


agg_data = pd.read_csv('telecom_data_with_pca.csv')  
cluster_descriptions = {0: "Low Engagement & Experience", 1: "High Engagement & Experience", 2: "Moderate Engagement & Experience"}  

less_engaged_cluster = 0
worst_experience_cluster = 0

engagement_scores = euclidean_distances(agg_data[['TCP DL Retrans. Vol (Bytes)', 'Avg RTT DL (ms)', 'Avg Bearer TP DL (kbps)']], KMeans.cluster_centers_[less_engaged_cluster].reshape(1, -1))
experience_scores = euclidean_distances(agg_data[['TCP DL Retrans. Vol (Bytes)', 'RTT', 'Avg Bearer TP DL (kbps)']], KMeans.cluster_centers_[worst_experience_cluster].reshape(1, -1))

satisfaction_scores = (engagement_scores + experience_scores) / 2

from sklearn.linear_model import LinearRegression

X = agg_data[['TCP_retransmission', 'RTT', 'Throughput']]
y = satisfaction_scores.reshape(-1, 1)

regression_model = LinearRegression()
regression_model.fit(X, y)

from sklearn.cluster import KMeans

satisfaction_scores_df = pd.DataFrame(satisfaction_scores, columns=['Satisfaction Score'])
kmeans = KMeans(n_clusters=2, random_state=42)
satisfaction_clusters = kmeans.fit_predict(satisfaction_scores_df)

clustered_data = agg_data.copy()
clustered_data['Satisfaction Cluster'] = satisfaction_clusters
avg_scores_per_cluster = clustered_data.groupby('Satisfaction Cluster').mean()

import mysql.connector

conn = mysql.connector.connect(host="localhost", user="admin", password="admin", database="aggregatedData")  

clustered_data.to_sql('satisfaction_scores', con=conn, if_exists='replace', index=False)

import datetime

model_info = {
    'code_version': 'v1.0',
    'start_time': datetime.datetime.now(),
    'end_time': None,
    'source': 'Linear Regression',
    'parameters': regression_model.get_params(),
    'metrics': regression_model.score(X, y),
    'artifacts': None  
}

model_info['end_time'] = datetime.datetime.now()

model_info_df = pd.DataFrame.from_dict(model_info, orient='index').T
model_info_df.to_csv('model_info.csv', index=False)
