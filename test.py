from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
# from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score
import pandas as pd
import os
import joblib

os.makedirs("model",exist_ok=True)

#Load
df=pd.read_csv("/Users/bhoomikasrivastava19/Documents/houseprice_predictor/house_prices.csv")

#Handling missing data
# print(df.isnull().sum())

df.drop(columns=['Title','Description','Price (in rupees)','Status','Transaction','Furnishing','facing','overlooking','Society','Car Parking','Ownership','Super Area','Dimensions','Plot Area'],inplace=True)
# print(df.columns)

df.info()

df['Carpet Area']=df['Carpet Area'].str.replace(',','',regex=False)
df['Carpet Area']=df['Carpet Area'].str.extract(r'(\d+\.?\d*)')

df['Carpet Area']=pd.to_numeric(df['Carpet Area'],errors='coerce')

df['Carpet Area'].fillna(df['Carpet Area'].mean(),inplace=True)

df['Bathroom']=pd.to_numeric(df['Bathroom'],errors='coerce')
df['Bathroom'].fillna(df['Bathroom'].median(),inplace=True)

df['Balcony']=pd.to_numeric(df['Balcony'],errors='coerce')
df['Balcony'].fillna(0,inplace=True)

df['Floor']=df['Floor'].str.extract(r'(\d+)')
df['Floor']=pd.to_numeric(df['Floor'],errors='coerce')
df['Floor'].fillna(0,inplace=True)
df=df.drop("Index",axis=1)

#print(df['Amount(in rupees)'].head)

def convert_price(x):
    x=str(x)

    if 'Cr' in x:
        return float(x.replace('Cr',''))*100
    if 'Lac' in x:
        return float(x.replace('Lac',''))
    
df['Amount(in rupees)']=df['Amount(in rupees)'].apply(convert_price)

print(df['Amount(in rupees)'].head)

df=df.dropna(subset=['Amount(in rupees)'])

print(df.isnull().sum())

df['price_per_sqft']=df['Amount(in rupees)']/df['Carpet Area']

df = df[df['price_per_sqft'] < df['price_per_sqft'].quantile(0.99)]
df = df[df['price_per_sqft'] > df['price_per_sqft'].quantile(0.01)]
df.drop(columns=['price_per_sqft'], inplace=True)


#Encoding for location

df=pd.get_dummies(df,columns=['location'])

#Scaling

model=Pipeline([
    ('scaler',StandardScaler()),
    ('regressor',RandomForestRegressor(n_estimators=10,random_state=42))
])

#split the data


X=df.drop(columns=['Amount(in rupees)'])
y=df['Amount(in rupees)']

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)

#Train
model.fit(X_train,y_train)

#predict
pred=model.predict(X_test)

#accuracy
print("Your accurancy:")
print(r2_score(y_test,pred))


joblib.dump(model,'model/houseprice.pkl')
joblib.dump(X.columns, "model/columns.pkl")

print("model saved!!!")

# print(X.columns)

