import numpy as np
import pandas as pd
import streamlit as st
from streamlit_folium import st_folium
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

header = st.container()
with header:
    st.title('Analysis dataframe of Appartement')
    st.markdown('''_we filtered our dataframe by Type of Appartement and scraped the link 
    to have more informations about the property sale_\n''')

st.sidebar.markdown("## Page 2")

st.markdown('\n')
st.markdown('\n')
st.markdown('\n')

# Load df of all real estate sales
df1 = pd.read_csv('Webscraping_DOMimmo_4.csv')

# Printing Dataframe of Appartement
st.markdown("**Let's take a look on the dataframe of Appartement :**")
st.dataframe(df1.head(9))
st.write('shape '+ str(np.shape(df1)))

st.markdown('\n')
st.markdown('\n')
st.markdown('\n')

# Heatmap to find relations between variables in a dataset.
st.markdown("<center><b>Let's find some relationship between variables :</b></center>", unsafe_allow_html=True)
fig = plt.figure(figsize=(10, 6))
fig.patch.set_facecolor('#0E1117')

df1_cor = df1.drop(['Address', 'Latitude', 'Longitude'], axis = 1)
ax = sns.heatmap(round(df1_cor.corr(numeric_only=True), 1), annot = True, cmap = 'magma', cbar = False)
cbar = plt.colorbar(ax.collections[0], ax=ax)
cbar.ax.tick_params(colors='white')
cbar.set_label('label', color='white')

plt.tick_params(colors='white', which='both')
st.write(fig)

st.markdown('\n')

st.markdown('''- we can see strong correlation between Number_Room vs Price (0.6)
- Very strong correlation between Number_Bedroom vs Number_Room (0.8)
- Strong correlation between Surface vs Number_Room (0.6)
- Strong correlation between Salles d'eau vs Toilettes (0.7)
- Strong correlation between Taxes foncière vs Salles d'eau (0.6)
''')

st.markdown('\n')
st.markdown('\n')

# Mean price by room 
apt_mean_price_by_room = df1.groupby(['Number_Room'])['Price (in €)'].mean().round().reset_index(name='Mean Price Room')
# Mean price by bedroom
apt_mean_price_by_bedroom = df1.groupby(['Number_Bedroom'])['Price (in €)'].mean().round().reset_index(name='Mean Price Bedroom')
# Mean price by Surface
apt_mean_price_by_surface = df1.groupby(['Surface (in m²)'])['Price (in €)'].mean().round().reset_index(name='Mean Price Surface')

st.markdown("**Let's see some relationships between Average price and some variables :**")
st.markdown('''_We use Scatter plot to observe linear relations between two variables in a dataset. 
In our case, price or mean price attribute is the dependent variable, 
and every other are the independent variables._''')

st.markdown('\n')
st.markdown('\n')

# Scatter plot Mean price by room 
fig = px.scatter(
    apt_mean_price_by_room, x='Number_Room', y='Mean Price Room', opacity=0.70,
    trendline='ols', trendline_color_override='darkblue'
)
fig.update_layout(title=dict(text='Mean Price of Apartments by Number of Rooms',
                            x=0.5, xanchor='center', y=0.9, yanchor='top',
                            font=dict(size=24, color='black')))
st.plotly_chart(fig)

# Scatter plot Mean price by bedroom
fig = px.scatter(
    apt_mean_price_by_bedroom, x = 'Number_Bedroom', y ='Mean Price Bedroom', opacity=0.70,
    trendline='ols', trendline_color_override='orange'
)
fig.update_layout(title=dict(text='Mean Price of Apartments by Number of Bedrooms',
                            x=0.5, xanchor='center', y=0.9, yanchor='top',
                            font=dict(size=24, color='black')))
st.plotly_chart(fig)

# Scatter plot Mean price by Surface
fig = px.scatter(
    apt_mean_price_by_surface, x = 'Surface (in m²)', y ='Mean Price Surface', opacity=0.70,
    trendline='ols', trendline_color_override='green'
)
fig.update_layout(title=dict(text='Mean Price of Apartments by Surface',
                            x=0.5, xanchor='center', y=0.9, yanchor='top',
                            font=dict(size=24, color='black')))
st.plotly_chart(fig)

st.markdown('\n')
st.markdown('\n')

# Mean price by city
apt_mean_price_by_city = df1.groupby(['City'])['Price (in €)'].mean().round().reset_index(name='Mean Price City')

colors = ['lightslategray',] * len(apt_mean_price_by_city)
max_price_idx = apt_mean_price_by_city['Mean Price City'].idxmax()
colors[max_price_idx] = 'crimson'

# Barplot of Mean price by City
fig = px.bar(apt_mean_price_by_city, y='Mean Price City', x='City', text_auto='.2s',
            title='Mean price of Appartement by City', color = colors)
fig.update_traces(textfont_size=12, textangle=0, textposition="outside", cliponaxis=False)
fig.update_layout(width=850, height=600)
st.plotly_chart(fig)