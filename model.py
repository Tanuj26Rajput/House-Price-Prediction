import pandas as pd
import numpy as np

house_df = pd.read_csv("House Price Prediction Dataset.csv")
house_df

print(house_df.isnull().sum())  # Shows the count of missing values per column
house_df = house_df.dropna()

def rmse(target, predictions):
    return np.sqrt(np.mean(np.square(target - predictions)))

from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

Garage_codes = {"Yes": 1, "No": 0}
house_df['Garage'] = house_df['Garage'].map(Garage_codes)

enc = preprocessing.OneHotEncoder()
enc.fit(house_df[['Location']])
enc.categories_
one_hot = enc.transform(house_df[['Location']]).toarray()
house_df[['Downtown', 'Rural', 'Suburban', 'Urban']] = one_hot

enc2 = preprocessing.OneHotEncoder()
enc2.fit(house_df[['Condition']])
enc2.categories_
one_hot = enc2.transform(house_df[['Condition']]).toarray()
house_df[['Excellent', 'Fair', 'Good', 'Poor']] = one_hot

numeric_cols = ['Area', 'Bedrooms', 'Bathrooms', 'Floors', 'YearBuilt']
scaler = StandardScaler()
scaler.fit(house_df[numeric_cols])

scaled_inputs = scaler.transform(house_df[numeric_cols])
cat_cols = ['Downtown', 'Rural', 'Suburban', 'Urban', 'Excellent', 'Fair', 'Good', 'Poor', 'Garage']
categorical_data = house_df[cat_cols].values

model = LinearRegression()
inputs = np.concatenate((scaled_inputs, categorical_data), axis = 1)
target = house_df['Price']
model.fit(inputs, target)
predictions = model.predict(inputs)
loss = rmse(target, predictions)
print("Loss: ", loss)

model.coef_

model.intercept_

from matplotlib import pyplot as plt

plt.scatter(house_df['YearBuilt'], target, color = "green", label = "Actual Price", alpha=0.6)
plt.scatter(house_df['YearBuilt'], predictions, color = "red", label = "Predicted Price", alpha=0.7)
plt.xlabel("Year Built")
plt.ylabel("Price")
plt.legend()
plt.show()

import joblib
joblib.dump(model, 'house_price_model.pkl')

joblib.dump(scaler, 'scaler.pkl')

joblib.dump(enc, 'location_encoder.pkl')
joblib.dump(enc2, 'condition_encoder.pkl')

print("Model and preprocessors saved successfully!")
