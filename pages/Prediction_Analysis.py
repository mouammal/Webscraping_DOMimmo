import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
import plotly.graph_objects as go

header = st.container()
with header:
    st.title('Prediction Price of Appartement')
    st.markdown('''_we cast all columns to numeric and encode categorical variables in one-hot encoding 
    to have more features for prediction_\n''')

st.sidebar.markdown("## Page 3")

st.markdown('\n')
st.markdown('\n')
st.markdown('\n')

# Load df of all real estate sales
df2 = pd.read_csv('Webscraping_DOMimmo_4.csv')

# Perform one hot encoding to convert categorical variables to binary variables.
df2 = pd.get_dummies(data = df2, columns = ['Environnement', 'Vue', 'Exposition']) 
df2.reset_index(drop=True, inplace=True)

# Printing cleaned and processed Dataframe of Appartement
st.markdown("**Let's take a look on the new procesed dataframe :**")
st.dataframe(df2.head(9))
st.write('shape '+ str(np.shape(df2)))

st.markdown('\n')
st.markdown('\n')
st.markdown('\n')

st.markdown("**Creating Train Data and Test Data :**")
st.markdown('''_it's important to rescale the variables so that they have a comparable scale. 
It's advised to use standardization or normalization so that the units of the coefficients 
obtained are all on the same scale._
''')

st.markdown('\n')

# Normalization
scaler = MinMaxScaler()

# Apply scaler() to all the columns except the 'dummy' variables
num_vars = ['Number_Room', 'Number_Bedroom', 'Surface (in m²)', 'Latitude',
       'Longitude', 'Surface séjour', 'Etage', "Nbre d'étages",
       'Salles de bain', 'Toilettes', 'Surface terrasse', 'Surface totale',
       'Taxe foncière', 'Surface terrain', "Salles d'eau", 'Taxe habitation', 'Price (in €)']

df2[num_vars] = scaler.fit_transform(df2[num_vars])

# Set our 'features' (x values) and our Y (target variable).
features = df2.drop(['Publishing Date', 'City', 'Address', 'Type', 
                                  'Description','Price (in €)', 'Latitude', 'Longitude',
                                  'Link', 'Link','Environnement_0', 'Vue_0','Exposition_0'], axis = 1) 
Y = df2['Price (in €)']

# Splitting feature and dependant variable Y (price) into train and test sets.
x_train, x_test, y_train, y_test = train_test_split(features, Y, test_size = 0.2, random_state = 0)
st.markdown('Shape of **x_train** :   '+ str(x_train.shape) + ' and **y_train** : '+ str(y_train.shape))
st.markdown('Shape of **x_test** :   '+ str(x_test.shape)+ ' and **y_test** : '+ str(y_test.shape))

st.markdown('\n')
st.markdown('\n')
st.markdown('\n')

st.markdown("### **Linear Regression Model using Sklearn**")
st.markdown('''_It's a supervised machine learning in which 
the predicted output is continuous with having a constant slope._
''')

# fit linear model 
LinearReg = LinearRegression().fit(x_train, y_train)

# prediction of linear model and print RMSE and R²
predictions = LinearReg.predict(x_test) 
LinearRegression_RMSE = np.sqrt(mean_squared_error(y_test, predictions))
LinearRegression_R2 = LinearReg.score(x_test, y_test)

st.write('RMSE Linear :', round(LinearRegression_RMSE, 4))
st.write('R2 Linear :', round(LinearRegression_R2, 4))

st.markdown('''_The R-squared (R2) value ranges from 0 to 1 with 1 defines perfect predictive accuracy.
RMSE is a measure of the difference between predicted values and actual values, from 0 to the maximum value of the dependent variable 
with 0 meaning that the predictions are perfect and there is no error in the model._
''')

# Regression plot with seaborn 
fig = plt.figure(figsize=(11, 6))
fig.patch.set_facecolor('#0E1117')
plt.rcParams.update({'axes.facecolor':'#0E1117'})

sns.regplot(x = y_test, y = predictions, data = df2, color = 'w')
fig.suptitle('y_test vs y_pred with Seaborn', color = 'w')  
plt.xlabel('y_test', color = 'white')                          
plt.ylabel('y_pred', color = 'white')
plt.tick_params(colors='white')
st.pyplot(fig) 

st.markdown('''**_PRICE_** = intercept +
        **coeff0** * _Number_Room_ + **coeff1** * _Number_Bedroom_ + **coeff2** * _Surface (in m²)_ + 
        **coeff3** * _Surface séjour_ + **coeff4** * _Etage_ + ...
        ''')

st.markdown('\n')
st.markdown('\n')
st.markdown('\n')

st.markdown("### **OLS Regression Model using Stat Models**")
st.markdown('''_OLS (Ordinary Least Squares) regression is a statistical method 
used to analyze the linear relationship between a dependent variable and one or more independent variables._
''')

# fit OLS model
ols = sm.OLS(y_train, x_train).fit()

# predict OLS model
pred = ols.predict(x_test)
OLSRegression_RMSE = np.sqrt(mean_squared_error(y_test, pred))
OLSRegression_R2 = ols.rsquared

st.write('RMSE OLS :', round(OLSRegression_RMSE, 4))
st.write('R2 OLS :', round(OLSRegression_R2, 4))

# Scatter plot with plotly express of OLS Model 
fig = px.scatter(
    df2, x = y_test, y = pred, opacity=0.70,
    trendline='ols', trendline_color_override='darkblue'
)
fig.update_layout(title=dict(text='y_test vs y_pred with Plotly Express',
                            x=0.2, xanchor='center', y=0.9, yanchor='top', font=dict(size=15, color='white')), 
                            xaxis_title="y_test", yaxis_title="y_pred")
st.plotly_chart(fig)

st.markdown('\n')
st.markdown('\n')
st.markdown('\n')

st.markdown("### **SVR Regression Model**")
st.markdown('''_Support Vector Regression (SVR) is a supervised ML algorithm for regression problems,
used for predicting continuous values rather than class labels or more independent variables.
In SVR, the goal is to find the optimal hyperplane that maximizes the margin, 
or the distance between the hyperplane and the closest data points (called support vectors)._
''')
st.markdown('''_SVR can be used for both linear and non-linear regression problems by using different kernel functions :_  
1. _SVR with Linear kernel_   
2. _SVR With Polynomial of degree 1 kernel_  
3. _SVR with Radial Basis Function (RBF) kernel_  
''') 

# fit SVR model
svr_lin  = SVR(kernel='linear', C=1e3).fit(x_train, y_train)
svr_poly = SVR(kernel='poly', C=1e3, degree=1).fit(x_train, y_train)
svr_rbf = SVR(kernel='rbf', C=1e3, gamma=0.1).fit(x_train, y_train)

# predict SVR model
pred_lin = svr_lin.predict(x_test)
pred_poly = svr_poly.predict(x_test)
pred_rbf = svr_rbf.predict(x_test)

st.markdown('\n')

col1, col2, col3 = st.columns(3)
with col1 : 
    svr_lin_RMSE = np.sqrt(mean_squared_error(y_test, pred_lin))
    svr_lin_R2 = svr_lin.score(x_test, y_test)
    st.write('RMSE SVM lin :', round(svr_lin_RMSE, 4))
    st.write('R2 SVM lin :', round(svr_lin_R2, 4))

with col2 :
    svr_poly_RMSE = np.sqrt(mean_squared_error(y_test, pred_poly))
    svr_poly_R2 = svr_poly.score(x_test, y_test)
    st.write('RMSE SVM poly :', round(svr_poly_RMSE, 4))
    st.write('R2 SVM poly :', round(svr_poly_R2, 4))

with col3 :
    svr_rbf_RMSE = np.sqrt(mean_squared_error(y_test, pred_rbf))
    svr_rbf_R2 = svr_rbf.score(x_test, y_test)
    st.write('RMSE SVM rbf :', round(svr_rbf_RMSE, 4))
    st.write('R2 SVM rbf :', round(svr_rbf_R2, 4))

st.markdown('\n')
st.markdown('\n')
st.markdown('\n')

st.markdown("### **Random Forest Regression Model**")
st.markdown('''_Random Forest Regression is a ML algorithm that utilizes multiple decision trees to make predictions. 
The idea behind this method is to create multiple decision trees using different subsets of the data and features, 
and then combine the predictions of these trees to make a final prediction._''')

# fit RF model
RF_reg = RandomForestRegressor(n_estimators=300).fit(x_train,y_train)

# predict RF model 
pred_RF = RF_reg.predict(x_test)

RF_RMSE = np.sqrt(mean_squared_error(y_test, pred_RF))
RF_R2 = RF_reg.score(x_test, y_test)
st.write('RMSE RF reg :', round(RF_RMSE, 4))
st.write('R2 RF reg :', round(RF_R2, 4))


st.markdown('\n')
st.markdown('\n')
st.markdown('\n')


st.markdown("### **Results of Best Model**")
st.markdown('''_We can see that the best model is OLS Regression Model with an R2_Score of {}%, 
followed by the RF regression model with {}%._  '''.format( round(OLSRegression_R2*100, 1), round(RF_R2*100, 1)) ) 

Model = ['Linear Regression', 'OLS Regression', 'SVM linear', 'SVM poly', 'SVM rbf', 'RF Regression']
Score = [LinearRegression_R2, OLSRegression_R2, svr_lin_R2, svr_poly_R2, svr_rbf_R2, RF_R2]
RMSE_model = [LinearRegression_RMSE, OLSRegression_RMSE, svr_lin_RMSE, svr_poly_RMSE, svr_rbf_RMSE, RF_RMSE]

# create a dataframe of all models and scores 
df_models = pd.DataFrame({'Model':Model, 'R2_Score':Score, 'RMSE': RMSE_model})

# bar plot of the dataframe showing the results
fig = go.Figure(data=[go.Bar(
    x=df_models['Model'],
    y=(df_models['R2_Score']).round(1),
    text=(df_models['R2_Score']*100).round(1).astype('str') + '%',
    textposition='auto',
    marker = {'color' : px.colors.diverging.Fall} # changing color of bar plot
)]) 
fig.update_layout(title='R2 score of all Models') 
fig.update_yaxes(tickformat=".0%")
st.plotly_chart(fig)


st.markdown('''_This results is obtained only with a dataframe of precise type (Appartement here) 
and thanks to webscraping a second time with the Link to have new informations._  ''')