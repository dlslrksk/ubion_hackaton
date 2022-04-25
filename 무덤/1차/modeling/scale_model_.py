from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression, Ridge, Lasso
import numpy as np

class complex():
    def __init__(self, data, target):
        self.data = data
        self.target = target
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(self.data, self.target,
                                    test_size=0.2,
                                    random_state=123)
        self.x_train = self.x_train
        self.x_test = self.x_test


    def scale_(self, data, scale):
        if scale == 'logscale':
            scaled = np.log1p(data)
            return scaled

        elif scale == 'standard':
            scaled = StandardScaler().fit_transform(data)
            return scaled

        elif scale == 'minmax':
            scaled = MinMaxScaler().fit_transform(data)
            return scaled


    def Regression_(self, model, n_estimators = 400, max_depth = 3, alpha = 1, max_iter = 1000, scaler = ""):

        if (model == 'LinearRegression') :
            if scaler:
                if (scaler == 'logscale'):
                    print(f"{model}\nAfter {scaler}")
                    x_train_scaled =  np.log(self.x_train)
                    x_test_scaled = np.log(self.x_test)
                    model = LinearRegression()
                    model.fit(x_train_scaled, self.y_train)
                    pred = model.predict(x_test_scaled)
                    mse = mean_squared_error(self.y_test, pred)
                    rmse = np.sqrt(mse)
                    r2 = r2_score(self.y_test, pred)
                    print(f'mse : {mse} \nrmse : {rmse} \nr2 : {r2} \ncoef : {model.coef_} \nintercept : {model.intercept_}')

                else:
                    print(f"{model}\nAfter {scaler}")
                    x_train_scaled =  scaler().fit_transform(self.x_train)
                    x_test_scaled = scaler().fit_transform(self.x_test)
                    model = LinearRegression()
                    model.fit(x_train_scaled, self.y_train)
                    pred = model.predict(x_test_scaled)
                    mse = mean_squared_error(self.y_test, pred)
                    rmse = np.sqrt(mse)
                    r2 = r2_score(self.y_test, pred)
                    print(f'mse : {mse} \nrmse : {rmse} \nr2 : {r2} \ncoef : {model.coef_} \nintercept : {model.intercept_}')
                
            else:
                print(f"{model}\nNO Scaler")
                model = LinearRegression()
                model.fit(self.x_train, self.y_train)
                pred = model.predict(self.x_test)
                mse = mean_squared_error(self.y_test, pred)
                rmse = np.sqrt(mse)
                r2 = r2_score(self.y_test, pred)
                print(f'mse : {mse} \nrmse : {rmse} \nr2 : {r2} \ncoef : {model.coef_} \nintercept : {model.intercept_}')

        if (model == 'Ridge') :
            if scaler:
                if (scaler == 'logscale'):
                    print(f"{model}\nAfter {scaler} with alpha : {alpha}")
                    x_train_scaled =  np.log(self.x_train)
                    x_test_scaled = np.log(self.x_test)
                    model = Ridge(alpha = alpha, max_iter=max_iter)
                    model.fit(x_train_scaled, self.y_train)
                    pred = model.predict(x_test_scaled)
                    mse = mean_squared_error(self.y_test, pred)
                    rmse = np.sqrt(mse)
                    r2 = r2_score(self.y_test, pred)
                    print(f'mse : {mse} \nrmse : {rmse} \nr2 : {r2} \ncoef : {model.coef_} \nintercept : {model.intercept_}')

                else:
                    print(f"{model}\nAfter {scaler}  with alpha : {alpha}")
                    x_train_scaled =  scaler().fit_transform(self.x_train)
                    x_test_scaled = scaler().fit_transform(self.x_test)
                    model = Ridge(alpha = alpha, max_iter = max_iter)
                    model.fit(x_train_scaled, self.y_train)
                    pred = model.predict(x_test_scaled)
                    mse = mean_squared_error(self.y_test, pred)
                    rmse = np.sqrt(mse)
                    r2 = r2_score(self.y_test, pred)
                    print(f'mse : {mse} \nrmse : {rmse} \nr2 : {r2} \ncoef : {model.coef_} \nintercept : {model.intercept_}')
                
            else:
                print(f"{model}\nNO Scaler with alpha : {alpha}")
                model = Ridge(alpha = alpha, max_iter = max_iter)
                model.fit(self.x_train, self.y_train)
                pred = model.predict(self.x_test)
                mse = mean_squared_error(self.y_test, pred)
                rmse = np.sqrt(mse)
                r2 = r2_score(self.y_test, pred)
                print(f'mse : {mse} \nrmse : {rmse} \nr2 : {r2} \ncoef : {model.coef_} \nintercept : {model.intercept_}')

        if (model == 'Lasso') :
            if scaler:
                if (scaler == 'logscale'):
                    print(f"{model}\nAfter {scaler} with alpha : {alpha}")
                    x_train_scaled =  np.log(self.x_train)
                    x_test_scaled = np.log(self.x_test)
                    model = Lasso(alpha = alpha, max_iter=max_iter)
                    model.fit(x_train_scaled, self.y_train)
                    pred = model.predict(x_test_scaled)
                    mse = mean_squared_error(self.y_test, pred)
                    rmse = np.sqrt(mse)
                    r2 = r2_score(self.y_test, pred)
                    print(f'mse : {mse} \nrmse : {rmse} \nr2 : {r2} \ncoef : {model.coef_} \nintercept : {model.intercept_}')

                else:
                    print(f"{model}\nAfter {scaler}  with alpha : {alpha}")
                    x_train_scaled =  scaler().fit_transform(self.x_train)
                    x_test_scaled = scaler().fit_transform(self.x_test)
                    model = Lasso(alpha = alpha, max_iter = max_iter)
                    model.fit(x_train_scaled, self.y_train)
                    pred = model.predict(x_test_scaled)
                    mse = mean_squared_error(self.y_test, pred)
                    rmse = np.sqrt(mse)
                    r2 = r2_score(self.y_test, pred)
                    print(f'mse : {mse} \nrmse : {rmse} \nr2 : {r2} \ncoef : {model.coef_} \nintercept : {model.intercept_}')
                
            else:
                print(f"{model}\nNO Scaler with alpha : {alpha}")
                model = Lasso(alpha = alpha, max_iter = max_iter)
                model.fit(self.x_train, self.y_train)
                pred = model.predict(self.x_test)
                mse = mean_squared_error(self.y_test, pred)
                rmse = np.sqrt(mse)
                r2 = r2_score(self.y_test, pred)
                print(f'mse : {mse} \nrmse : {rmse} \nr2 : {r2} \ncoef : {model.coef_} \nintercept : {model.intercept_}')


        if (model == 'RandomForest') :
            if scaler:
                if (scaler == 'logscale'):
                    print(f"{model}\nAfter {scaler}")
                    x_train_scaled =  np.log(self.x_train)
                    x_test_scaled = np.log(self.x_test)
                    model = RandomForestRegressor(n_estimators = n_estimators, max_depth = max_depth)
                    model.fit(x_train_scaled, self.y_train)
                    pred = model.predict(x_test_scaled)
                    mse = mean_squared_error(self.y_test, pred)
                    rmse = np.sqrt(mse)
                    r2 = r2_score(self.y_test, pred)
                    print(f'mse : {mse} \nrmse : {rmse} \nr2 : {r2}')

                else:
                    print(f"{model}\nAfter {scaler}")
                    x_train_scaled =  scaler().fit_transform(self.x_train)
                    x_test_scaled = scaler().fit_transform(self.x_test)
                    model = RandomForestRegressor(n_estimators = n_estimators, max_depth = max_depth)
                    model.fit(x_train_scaled, self.y_train)
                    pred = model.predict(x_test_scaled)
                    mse = mean_squared_error(self.y_test, pred)
                    rmse = np.sqrt(mse)
                    r2 = r2_score(self.y_test, pred)
                    print(f'mse : {mse} \nrmse : {rmse} \nr2 : {r2}')
                
            else:
                print(f"{model}\nNO Scaler")
                model = RandomForestRegressor(n_estimators = n_estimators, max_depth = max_depth)
                model.fit(self.x_train, self.y_train)
                pred = model.predict(self.x_test)
                mse = mean_squared_error(self.y_test, pred)
                rmse = np.sqrt(mse)
                r2 = r2_score(self.y_test, pred)
                print(f'mse : {mse} \nrmse : {rmse} \nr2 : {r2}')    
