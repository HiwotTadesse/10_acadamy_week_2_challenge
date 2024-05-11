import pandas as pd
import psycopg2
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import sys



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

agg_data = telecom_data.groupby('MSISDN/Number').agg({
    'TCP DL Retrans. Vol (Bytes)': lambda x: x.fillna(x.mean()),
    'TCP UL Retrans. Vol (Bytes)': lambda x: x.fillna(x.mean()),
    'Avg RTT DL (ms)': lambda x: x.fillna(x.mean()),
    'Avg RTT UL (ms)': lambda x: x.fillna(x.mean()),
    'Handset Type': lambda x: x.mode()[0],
    'Avg Bearer TP DL (kbps)': lambda x: x.fillna(x.mean()),
    'Avg Bearer TP UL (kbps)': lambda x: x.fillna(x.mean()),
})

top_TCP_values = telecom_data['TCP DL Retrans. Vol (Bytes)'].nlargest(10)
bottom_TCP_values = telecom_data['TCP DL Retrans. Vol (Bytes)'].nsmallest(10)
most_freq_TCP_values = telecom_data['TCP DL Retrans. Vol (Bytes)'].mode()

top_RTT_values = telecom_data['Avg RTT DL (ms)'].nlargest(10)
bottom_RTT_values = telecom_data['Avg RTT DL (ms)'].nsmallest(10)
most_freq_RTT_values = telecom_data['Avg RTT DL (ms)'].mode()

top_throughput_values = telecom_data['Avg Bearer TP DL (kbps)'].nlargest(10)
bottom_throughput_values = telecom_data['Avg Bearer TP DL (kbps)'].nsmallest(10)
most_freq_throughput_values = telecom_data['Avg Bearer TP DL (kbps)'].mode()

average_throughput_per_handset = agg_data.groupby('Handset Type')['Avg Bearer TP DL (kbps)'].mean()
average_TCP_retransmission_per_handset = agg_data.groupby('Handset Type')['TCP DL Retrans. Vol (Bytes)'].mean()

scaler = StandardScaler()
scaled_data = scaler.fit_transform(agg_data[['TCP DL Retrans. Vol (Bytes)', 'Avg RTT DL (ms)', 'Avg Bearer TP DL (kbps)']])

kmeans = KMeans(n_clusters=3, random_state=42)
clusters = kmeans.fit_predict(scaled_data)

agg_data['Cluster'] = clusters

cluster_descriptions = {
    0: "High TCP retransmission, high RTT, low throughput",
    1: "Low TCP retransmission, low RTT, high throughput",
    2: "Moderate TCP retransmission, moderate RTT, moderate throughput"
}
