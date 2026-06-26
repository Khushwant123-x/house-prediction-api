from sklearn.datasets import fetch_california_housing
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error,r2_score
import pandas as pd
import joblib

data = fetch_california_housing()

x= pd.DataFrame(data.data,columns=data.feature_names)
y= data.target

#print("total records:",{x.shape[0]})

X_train,X_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=42)

# training model
model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

model.fit(X_train,y_train)

y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test,y_pred)

r2 = r2_score(y_test,y_pred)

print(f"average error: ${mae*100000:,.0f}")

joblib.dump(model,"house_model.joblib")
joblib.dump(list(x.columns),"house_features.joblib")
