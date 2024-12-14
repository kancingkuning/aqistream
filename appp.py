import pandas as pd
import numpy as np
import datetime
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
sns.set_theme(style='whitegrid', context='paper')


#judul
st.header("How Air Pollution Affects Stations Around Beijing :cn: :dash:")


#load data, buat rentang waktu yang bisa diklik
all_df = pd.read_csv("combined_file.csv")
datetime_columns = ["datetime"]
all_df.sort_values(by="datetime", inplace=True)
all_df.reset_index(inplace=True)

for column in datetime_columns:
    all_df[column] = pd.to_datetime(all_df[column])
start_date = all_df['datetime'].min()
end_date = all_df['datetime'].max()

#sidebar untuk rentang waktu yg bisa diklik
with st.sidebar:
    # Mengambil start_date & end_date dari date_input
    start_date, end_date = st.date_input(
        label='Timerange',min_value=start_date,
        max_value=end_date,
        value=[start_date, end_date]
    )

main_df = all_df[(all_df["datetime"] >= str(start_date)) & 
                (all_df["datetime"] <= str(end_date))]
stations = main_df['station'].unique()
selected_stations = st.sidebar.multiselect('Select Stations',
                                           options=stations,
                                           default=stations.tolist())

filtered_df = main_df[main_df['station'].isin(selected_stations)]

#buat fungsi untuk chart distribusi PM
def create_pollutant_distribution(all_df):
   pollutant_distribution_df = all_df.resample(rule='D', on='datetime').agg({
        "PM2.5": "sum",
        "PM10": "sum"
    }).reset_index()

   return pollutant_distribution_df

# Membuat pengelompokan Air Quality Index berdasarkan data PM2.5
def calculate_aqi_pm25(pm25):
    if pm25 <= 12:
        return (pm25 / 12) * 50  # Good
    elif pm25 <= 35.4:
        return ((pm25 - 12.1) / (35.4 - 12.1)) * 50 + 50  # Moderate
    elif pm25 <= 55.4:
        return ((pm25 - 35.5) / (55.4 - 35.5)) * 50 + 100  # Unhealthy for Sensitive Groups
    elif pm25 <= 150.4:
        return ((pm25 - 55.5) / (150.4 - 55.5)) * 100 + 150  # Unhealthy
    elif pm25 <= 250.4:
        return ((pm25 - 150.5) / (250.4 - 150.5)) * 100 + 200  # Very Unhealthy
    elif pm25 > 250.4:
        return 300  # Hazardous
    else:
        return None  # Missing data


filtered_df['AQI'] = filtered_df['PM2.5'].apply(calculate_aqi_pm25)

def categorize_aqi(aqi):
    if aqi is None:
        return "Unknown"
    elif aqi <= 50:
        return "Good"
    elif aqi <= 100:
        return "Moderate"
    elif aqi <= 150:
        return "Unhealthy for Sensitive Groups"
    elif aqi <= 200:
        return "Unhealthy"
    elif aqi <= 300:
        return "Very Unhealthy"
    else:
        return "Hazardous"


filtered_df['AQI_Category'] = filtered_df['AQI'].apply(categorize_aqi)


def create_aqi_category_count(filtered_df):
    aqi_category_count_df = filtered_df['AQI_Category'].value_counts().reset_index()
    aqi_category_count_df.columns = ['AQI_Category', 'Count']
    return aqi_category_count_df

def create_goodaqi_stations_percentage(filtered_df):
    stations_with_good_AQI = filtered_df[filtered_df['AQI_Category'] == 'Good']
    stations_count = stations_with_good_AQI['station'].value_counts()
    total_good_occurrences = stations_count.sum() 
    goodaqi_stations_percentage = (stations_count / total_good_occurrences) * 100
    return goodaqi_stations_percentage


weather_columns = ['TEMP', 'PRES', 'DEWP', 'DEWP', 'RAIN', 'WSPM']

def create_aqi_weather_cor(filtered_df, weather_columns):
   aqi_weather_cor_df = filtered_df.resample(rule='D', on='datetime').agg({
        "AQI": "mean",
        **{col: "mean" for col in weather_columns} 
    }).reset_index()
   
   correlation_matrix = aqi_weather_cor_df.corr()
   return correlation_matrix



pollutant_distribution_df = create_pollutant_distribution(filtered_df)
aqi_category_count_df = create_aqi_category_count(filtered_df)
correlation_matrix = create_aqi_weather_cor(filtered_df, weather_columns)
goodaqi_stations_percentage = create_goodaqi_stations_percentage(filtered_df)

#chart 1 untuk PM2.5 dan PM10
st.subheader('PM Daily Distribution')
st.write('PM stands for Particulate Matters, refers to tiny solid particles or liquid droplets suspended in the air. \n This particle can be harmful to your health, especially around busy places like train stations.')
st.write('PM2.5: These are super tiny particles, about 30 times smaller than a human hair. Because of their size, they can get deep into your lungs and even your bloodstream, causing problems like breathing issues or heart disease.')
st.write('PM10: These are bigger than PM2.5 but still small enough to get into your nose and lungs. They can cause irritation, coughing, or worsen allergies.')

#buat kolom untuk chart supaya bisa kiri kanan
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(35, 15))


sns.histplot(pollutant_distribution_df['PM2.5'], bins=30, kde=True, color='#dcd0ff', ax=ax[0])
ax[0].set_title('Distribution of PM2.5', fontsize=50)
ax[0].set_xlabel('PM2.5 Levels', fontsize=30)
ax[0].set_ylabel('Frequency', fontsize=30)
ax[0].tick_params(axis='y', labelsize=25)
ax[0].tick_params(axis='x', labelsize=25)


sns.histplot(pollutant_distribution_df['PM10'], bins=30, kde=True, color='#040f56', ax=ax[1])
ax[1].set_title('Distribution of PM10', fontsize=50)
ax[1].set_xlabel('PM10 Levels', fontsize=30)
ax[1].set_ylabel('Frequency', fontsize=30)
ax[1].tick_params(axis='y', labelsize=25)
ax[1].tick_params(axis='x', labelsize=25)

#atur ukuran layout
fig.tight_layout(pad=10)

#untuk memunculkan di streamlit
st.pyplot(fig)

#chart 3 bar category
st.subheader("Count of Air Quality Index based on Category")
st.write("The Air Quality Index (AQI) tells us how clean or polluted the air is and how it might affect our health.")
fig, ax = plt.subplots(figsize=(12, 8))  
num_categories = len(filtered_df['AQI_Category'].unique()) 
palette = [mpl.cm.twilight_shifted(i / num_categories) for i in range(num_categories)]
p = sns.countplot(
    x='AQI_Category',  # Corrected to use the 'AQI_Category' column
    data=filtered_df,   # Pass the filtered dataframe
    order=filtered_df['AQI_Category'].value_counts().index, 
    palette=palette,  
    ax=ax            
)
for container in p.containers:
        p.bar_label(container, label_type='center', fontsize=14, color='white')  # Add the counts inside bars
    
p.set_title('AQI Category Count', fontsize=20)
p.set_xlabel('AQI Category', fontsize=15)
p.set_ylabel('Count', fontsize=15)
p.tick_params(axis='y', labelsize=12)
p.tick_params(axis='x', labelsize=12)
plt.xticks(rotation=45, ha='right')
st.pyplot(fig)

# chart 4 bar good aqi
st.subheader("Good Air Quality Index based on Stations")
st.write("Desirable Air Quality Percentage in each stations.")
fig, ax = plt.subplots(figsize=(12, 8)) 
sns.barplot(x=goodaqi_stations_percentage.index, y=goodaqi_stations_percentage.values, color='#dcd0ff', ax=ax)
ax.set_title("Percentage of 'Good' AQI Occurrences per Station", fontsize=20)
ax.set_xlabel('Stations', fontsize=15)
ax.set_ylabel('Percentage (%)', fontsize=15)
ax.tick_params(axis='y', labelsize=12)
ax.tick_params(axis='x', labelsize=12)
ax.tick_params(axis='x', rotation=90)
fig.tight_layout(pad=10)
st.pyplot(fig)

#heatmap
st.subheader("Correlation Matrix of AQI Category and Weather Factors")
st.write(" the relationships between air quality levels (AQI) and various weather conditions like temperature, humidity, wind speed, and pressure.")
fig, ax = plt.subplots(figsize=(12, 8))
sns.heatmap(
    correlation_matrix, 
    annot=True, 
    cmap='twilight_shifted', 
    fmt='.2f', 
    vmin=-1, 
    vmax=1, 
    ax=ax
)
st.pyplot(fig)
