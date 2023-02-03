import numpy as np
import pandas as pd
import streamlit as st
from streamlit_folium import st_folium
import folium

header = st.container()
with header:
    st.title('DOMimmo Webscraping')
    st.markdown('_WebScraping of All real estate sales in Martinique, visualization, analysis and price prediction_\n')

st.sidebar.markdown("## Main Page")
st.sidebar.text("This main page's about scraping\n" 
"DOMimmo webpage, retrieving\n"
"data to put them in a dataframe\n"
"filtering & downloading it\n"
"as a CSV file.") 
st.sidebar.text("We display descriptive staticts\n"
"about dataframe & key-numbers\n"
"on average price of appartment,\n"
"maison/villa or terrain.\n")
st.sidebar.text("We also display maps of\n"
"Martinique showing different\n"
"informations by city.")
st.sidebar.text('\n')

st.markdown('\n')
st.markdown('\n')
st.markdown('\n')

@st.cache
def convert_df(df):
    # IMPORTANT: Cache the conversion to prevent computation on every rerun
    return df.to_csv().encode('utf-8')

# Load df of all real estate sales
df = pd.read_csv('Webscraping_DOMimmo_2.csv')

col1, col2 = st.columns(2)
with col1 : 
    # Multiselect by Type
    type_option = st.multiselect('Type of real estate ', df.Type.unique())
    df_type = df[df['Type'].isin(type_option)]

    # Select min Price
    min_price = st.selectbox('Min price', ([0, 25000, 50000, 75000] + list(np.arange(100000, 5050000, 150000))), 
                             format_func = lambda x: "{:,} €".format(x))

with col2 : 
    # Multiselect by City
    city_option = st.multiselect(' City ', df.City.unique())
    df_city = df[df['City'].isin(city_option)]

    # Select min Price
    max_price = st.selectbox('Max price', (([25000, 50000, 75000] + list(np.arange(100000, 5050000, 150000)))[::-1]), 
                             format_func = lambda x: "{:,} €".format(x))

# Slider Select Surface
surface_option = st.slider('Select a range of Surface in m²', 0, 2500, (0, 2500), step = 10)
st.write('from  ', surface_option[0], ' m²  to  ', surface_option[1], ' m²')


# Dataframe of Price and Surface
df_price = df[ (df['Price (in €)'] >= min_price) & (df['Price (in €)'] <= max_price)]
df_surface = df[ (df['Surface (in m²)'] >= surface_option[0]) & (df['Surface (in m²)'] <= surface_option[1])]

st.markdown('\n')
st.markdown('\n')
st.markdown('\n')

# Printing filtered Dataframe
st.markdown("**Let's take a look on data :**")
if len(type_option) and len(city_option) != 0:
    df_merged_tc = pd.merge(df_type, df_city) # tc = type-city
    df_merged_tcp = pd.merge(df_merged_tc, df_price) # tcp = type-city-price
    st.dataframe(pd.merge(df_merged_tcp, df_surface))
    st.write('shape : '+str(np.shape(pd.merge(df_merged_tcp, df_surface))))

    # Download Button
    st.sidebar.download_button( label="Download fitered dataframe as CSV",
                        data = convert_df(pd.merge(df_merged_tcp, df_surface)),
                        file_name ='new_df_dom.csv',
                        mime = 'text/csv',
                    )

elif len(type_option) != 0:
    df_merged_tp = pd.merge(df_type, df_price) # tp = type-price 
    st.dataframe(pd.merge(df_merged_tp, df_surface))
    st.write('shape '+str(np.shape(pd.merge(df_merged_tp, df_surface))))

    # Download Button
    st.sidebar.download_button( label="Download fitered dataframe as CSV",
                        data = convert_df(pd.merge(df_merged_tp, df_surface)),
                        file_name ='new_df_dom.csv',
                        mime = 'text/csv',
                    )
    
elif len(city_option) != 0:
    df_merged_cp = pd.merge(df_city, df_price) # cp = city-price 
    st.dataframe(pd.merge(df_merged_cp, df_surface))
    st.write('shape '+str(np.shape(pd.merge(df_merged_cp, df_surface))))
    
    # Download Button
    st.sidebar.download_button( label="Download fitered dataframe as CSV",
                        data = convert_df(pd.merge(df_merged_cp, df_surface)),
                        file_name ='new_df_dom.csv',
                        mime = 'text/csv',
                    )

else :
    st.dataframe(pd.merge(df_price, df_surface))
    st.write('shape '+str(np.shape(pd.merge(df_price, df_surface))))
    
    # Download Button
    st.sidebar.download_button( label="Download fitered dataframe as CSV",
                        data = convert_df(pd.merge(df_price, df_surface)),
                        file_name ='new_df_dom.csv',
                        mime = 'text/csv',
                    )

st.markdown('\n')
st.markdown('\n')
st.markdown('\n')

# Print average price of maison/villa, appartement & terrain
col1, col2, col3 = st.columns(3)
with col1:
    avg_surface = str(round(df[df['Type'] == 'Maison / Villa']['Price (in €)'].mean())//round(df[df['Type'] == 'Maison / Villa']['Surface (in m²)'].mean()))
    avg_maison = "{:,}".format(round(df[df['Type'] == 'Maison / Villa']['Price (in €)'].mean()))
    st.metric(label = "Average Maison / Villa", value = avg_maison + " €", delta = avg_surface + " €/m²")

with col2:
    avg_surface = str(round(df[df['Type'] == 'Appartement']['Price (in €)'].mean())//round(df[df['Type'] == 'Appartement']['Surface (in m²)'].mean()))
    avg_apt = "{:,}".format(round(df[df['Type'] == 'Appartement']['Price (in €)'].mean()))
    st.metric(label = "Average Appartement", value = avg_apt + " €", delta = avg_surface + " €/m²")

with col3:
    avg_surface = str(round(df[df['Type'] == 'Terrain']['Price (in €)'].mean())//round(df[df['Type'] == 'Terrain']['Surface (in m²)'].mean()))
    avg_terrain = "{:,}".format(round(df[df['Type'] == 'Terrain']['Price (in €)'].mean()))
    st.metric(label = "Average Terrain", value = avg_terrain + " €", delta = avg_surface + " €/m²")

st.markdown('\n')
st.markdown('\n')
st.markdown('\n')

# Print staticts about all data of dataframe 
st.markdown("**Descriptive statistics about all data :**")
st.write(df.describe().drop(['Address', 'Latitude', 'Longitude'], axis=1).style.format("{:.0f}"))

st.markdown('\n')
st.markdown('\n')
st.markdown('\n')

# Print Map of Martinique 
col1, col2 = st.columns(2)
with col1 : 
    # Start location is MARTINIQUE
    martinique_coordinate = [14.641528, -61.024174]
    map = folium.Map(location = martinique_coordinate, zoom_start = 10)

    # dataframe of number of property sales by city, lat and long
    df_count_city = df.groupby(['City', 'Latitude', 'Longitude']).size().reset_index(name='counts')
    df_count_city.sort_values(by='counts', ascending=False, inplace=True)

    occurences = folium.map.FeatureGroup()
    n_mean = df_count_city['counts'].mean()
    for lat, long, number, city in zip(df_count_city['Latitude'],
                                   df_count_city['Longitude'],
                                   df_count_city['counts'],
                                   df_count_city['City'], ):
        occurences.add_child(
        folium.vector_layers.CircleMarker( [lat, long],
                                       radius = number/n_mean*5, # define how big you want circle markers to be
                                       color='red',
                                       fill=True,
                                       fill_color='blue',
                                       fill_opacity=0.4,
                                       tooltip = str(city) +', '+ str(number)
                                     )
        )
    st.markdown('**Map of Count property by City in Martinique**')
    st_folium(map.add_child(occurences), width=400, height=500)

    st.markdown('''
    - _Most of property sale are located in South of the islands and near coast._
    - _Especially in South-West, on the side of the carribean sea where Fort-de-France has the most property sales._
    ''')


with col2 : 
    # Start location is MARTINIQUE
    martinique_coordinate = [14.641528, -61.024174]
    map = folium.Map(location = martinique_coordinate, zoom_start=10)

    # dataframe of mean price by city, lat and long
    df_mean_city = df.groupby(['City', 'Latitude', 'Longitude'])['Price (in €)'].mean().round().reset_index(name='Mean Price')
    df_mean_city.sort_values(by='Mean Price', ascending=False, inplace=True)

    occurences = folium.map.FeatureGroup()
    total_mean = df_mean_city['Mean Price'].mean()
    for lat, long, mean_p, city in zip(df_mean_city['Latitude'],
                                   df_mean_city['Longitude'],
                                   df_mean_city['Mean Price'],
                                   df_mean_city['City'], ):
        occurences.add_child(
        folium.vector_layers.CircleMarker( [lat, long],
                                       radius = mean_p/total_mean*5, # define how big you want circle markers to be
                                       color='blue',
                                       fill=True,
                                       fill_color='green',
                                       fill_opacity=0.4,
                                       tooltip = str(city) +', avg price = '+ str(mean_p) + '€'
                                     )
        )
    st.markdown('**Map of Average Price by City in Martinique**')
    st_folium(map.add_child(occurences), width=400, height=500)
    
    st.markdown('''
    - _Property sale that are located on South-West of the island on the carrabean sea cost almost 1M€._  
    - _The lowest average price for the sale of a property is located inland._  
    - _The average price of a property sale is increasing wheter we are near a coast._  
    - _Fort-de-France, the 'chef-lieu' of Martinique has a low avg price around 300k €._  
    ''')
