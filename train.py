import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import pickle

df = pd.read_csv('data/yield_df.csv')

# Clean
df = df.dropna()

# Select columns
df = df[['Area', 'Item', 'Year',
         'average_rain_fall_mm_per_year',
         'pesticides_tonnes',
         'avg_temp',
         'hg/ha_yield']]

# One-hot encoding
df = pd.get_dummies(df, columns=['Area', 'Item'])

# Features & target
X = df.drop('hg/ha_yield', axis=1)
y = df['hg/ha_yield']

# Save columns
model_columns = X.columns

# Train
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = RandomForestRegressor()
model.fit(X_train, y_train)

# Save model
pickle.dump(model, open('model/model.pkl', 'wb'))
pickle.dump(model_columns, open('model/columns.pkl', 'wb'))

print("Final model trained!")