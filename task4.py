import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import geopandas as gpd
import folium
from folium.plugins import HeatMap
from sklearn.cluster import KMeans

data = pd.read_csv('C:/Users/sonali gupta31102003/Downloads/Road Accident Data.csv')
print(data.columns)
data.columns = data.columns.str.strip()
data['Accident Date'] = pd.to_datetime(data['Accident Date'], errors='coerce')
data['Time'] = pd.to_datetime(data['Time'], format='%H:%M', errors='coerce').dt.time
data['Hour'] = pd.to_datetime(data['Time'], format='%H:%M:%S', errors='coerce').dt.hour
data['DayOfWeek'] = data['Accident Date'].dt.day_name()
plt.figure(figsize=(10, 6))
sns.histplot(data['Hour'].dropna(), bins=24, kde=False, color='blue')
plt.title('Accidents by Time of Day')
plt.xlabel('Hour of the Day')
plt.ylabel('Number of Accidents')
plt.xticks(range(24))
plt.show()
plt.figure(figsize=(10, 6))
sns.countplot(data=data, x='Weather_Conditions', order=data['Weather_Conditions'].value_counts().index)
plt.title('Accidents by Weather Condition')
plt.xticks(rotation=45)
plt.ylabel('Number of Accidents')
plt.show()
plt.figure(figsize=(10, 6))
sns.countplot(data=data, x='Road_Surface_Conditions', order=data['Road_Surface_Conditions'].value_counts().index)
plt.title('Accidents by Road Surface Condition')
plt.xticks(rotation=45)
plt.ylabel('Number of Accidents')
plt.show()
if data['Latitude'].notnull().all() and data['Longitude'].notnull().all():
    accident_map = folium.Map(location=[data['Latitude'].mean(), data['Longitude'].mean()], zoom_start=12)
    heat_data = [[row['Latitude'], row['Longitude']] for index, row in data.iterrows() if not pd.isnull(row['Latitude']) and not pd.isnull(row['Longitude'])]
    HeatMap(heat_data, radius=10, blur=15, max_zoom=1).add_to(accident_map)
    accident_map.save('accident_hotspots.html')
coords = data[['Latitude', 'Longitude']].dropna()
if not coords.empty:
    kmeans = KMeans(n_clusters=10, random_state=0).fit(coords)
    data.loc[coords.index, 'Cluster'] = kmeans.labels_
    cluster_map = folium.Map(location=[coords['Latitude'].mean(), coords['Longitude'].mean()], zoom_start=12)
    colors = ['blue', 'red', 'green', 'purple', 'orange', 'darkred', 'lightred', 'beige', 'darkblue', 'darkgreen']
    
    for index, row in data.dropna(subset=['Cluster']).iterrows():
        folium.CircleMarker(
            [row['Latitude'], row['Longitude']],
            radius=5,
            color=colors[int(row['Cluster'])],
            fill=True,
            fill_color=colors[int(row['Cluster'])],
            fill_opacity=0.6
        ).add_to(cluster_map)

    cluster_map.save('accident_clusters.html')
