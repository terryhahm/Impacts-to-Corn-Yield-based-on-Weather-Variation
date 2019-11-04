import pandas as pd
import numpy as np
import importlib
from string import ascii_letters
import matplotlib.pyplot as plt
from matplotlib.pylab import rcParams
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# test

#### PRE_PROCESSING

# Reading data from csv  files and merging feature

features = pd.read_csv('data.csv')

features[['year', 'month', 'day']] = features[['year', 'month', 'day']].astype(str)

features['date'] = features['month'] + "-" + features['day'] + "-" + features['year']

features.drop(columns = ['year', 'month', 'day', 'Unnamed: 29'], axis=1, inplace=True)

features.to_csv('new_raw_data.csv')

new_headers = ['date', 'avg_wind_speed', 'avg_wind_dir', 'sol_rad', 'avg_air_temp', 'avg_rel_hum', 'avg_dewpt_temp', 'precip', 'pot_evapot', 'avg_soiltemp_4in_sod', 'site']

new_features = features[new_headers].copy(deep = True)

pivot_features = ['avg_wind_speed', 'avg_wind_dir', 'sol_rad', 'avg_air_temp', 'avg_rel_hum', 'avg_dewpt_temp', 'precip', 'pot_evapot', 'avg_soiltemp_4in_sod']

new_features_pivot = new_features.pivot(index = 'site', columns = 'date', values = pivot_features)

new_features_pivot.columns = list(map("_".join, list(new_features_pivot.columns)))

new_features_pivot.to_csv('new_transformed_data_2.csv')

new_features_pivot.reset_index()

feature_columns = new_features_pivot.columns

new_features_pivot[feature_columns] = new_features_pivot[feature_columns].apply(pd.to_numeric, errors='coerce')

new_features_pivot.to_csv('new_transformed_data_3.csv')

new_features_pivot.fillna(0)

corn_yield = [209.4, 170.3, 206.8, 206.8, 204.1, 151.9, 157.6, 206, 165.4, 227.9, 224.4, 246.7, 148, 191, 127.9, 166.7, 201.8, 212.4, 176.4]

#### Penalized Regression model

X = new_features_pivot.values

X.astype('float64')

X[np.isnan(X)] = 0

X[np.isnan(X)] = 0

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

y = corn_yield

### FUNCTIONAL FORM 

def ridge_regression(X, y, alpha,mi, mx, models_to_plot={}):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, shuffle = True )
    ridgereg = Ridge(alpha=alpha, fit_intercept=True, normalize=False, tol = 0.0001, solver = 'svd')
    ridgereg.fit(X_train, y_train)
    y_pred = ridgereg.predict(X_test)

    x_trend = np.arange(mi,mx,1)
    y_trend = x_trend

    if alpha in models_to_plot:
        plt.subplot(models_to_plot[alpha])
        plt.tight_layout()
        plt.xlim(mi,mx)
        plt.ylim(mi,mx)
        plt.plot(y_test,y_pred,'.')
        plt.plot(x_trend,y_trend,'-')
        plt.title('Plot for alpha: %.3g'%alpha)

    rss = sum((y_pred-y_test)**2)
    ret = [rss]
    ret.extend([ridgereg.score(X_test,y_test)])
    ret.extend([ridgereg.intercept_])
    ret.extend(ridgereg.coef_)
    return ret

alpha_ridge = [1e-15, 1e-10, 1e-8, 1e-4, 1e-3,1e-2, 1, 5, 10, 20]

models_to_plot = {1e-15:231, 1e-10:232, 1e-4:233, 1e-3:234, 1e-2:235, 5:236}

#Initialize the dataframe for storing coefficients.
col = ['rss','score','intercept'] + [i for i in list(new_features_pivot)]
ind = ['alpha_%.2g'%alpha_ridge[i] for i in range(len(alpha_ridge))]
coef_matrix_ridge = pd.DataFrame(index=ind, columns=col)

for i in range(len(alpha_ridge)):
    coef_matrix_ridge.iloc[i,] = ridge_regression(X_scaled, y, alpha_ridge[i], 100, 250, models_to_plot)

coef_matrix_ridge.to_csv('Ridge_Regression_coefficients.csv')
plt.show()

