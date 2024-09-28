import pickle
import pandas as pd
import os
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


class ModelTrainer:
    def __init__(self, save_path='best_model.pkl'):
        self.save_path = save_path
        self.best_accuracy = float('-inf')
        self.best_model = None

    def performance_metrics(self, y_true, y_pred, model_name, n_predictors):
        mse = mean_squared_error(y_true, y_pred)
        rmse = mse ** 0.5
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)

        # Adjusted R² calculation
        n = len(y_true)
        adjusted_r2 = 1 - (1 - r2) * (n - 1) / (n - n_predictors - 1)

        return pd.DataFrame({
            'Model': [model_name],
            'MSE': [mse],
            'RMSE': [rmse],
            'MAE': [mae],
            'R2': [r2],
            'Adjusted R2': [adjusted_r2]
        })

    def train_and_save_best_model(self, X_train, y_train, X_test, y_test):
        dfOne = pd.DataFrame()

        # Example usage:
        models = {
            'Linear Regression': [LinearRegression(), {}],
            'Ridge Regression': [Ridge(), {'alpha': [0.1, 1, 10], 'fit_intercept': [True], 'max_iter': [None]}],
            'Lasso Regression': [Lasso(), {'alpha': [0.1, 1, 10], 'fit_intercept': [True], 'max_iter': [1000]}],
            'Elastic Net': [ElasticNet(), {'alpha': [0.1, 1, 10], 'l1_ratio': [0.5], 'fit_intercept': [True], 'max_iter': [1000]}]
        }

        for model_name, model_instance in models.items():
            model = model_instance[0]
            params = model_instance[1]
            
            # Perform Grid Search
            grid_search = GridSearchCV(model, params, cv=5, n_jobs=-1, verbose=2)
            grid_search.fit(X_train, y_train)

            best_model_instance = grid_search.best_estimator_
            y_pred = best_model_instance.predict(X_test)
            n_predictors = X_train.shape[1]

            # Calculate performance metrics
            df = self.performance_metrics(y_test, y_pred, model_name,n_predictors)
            dfOne = pd.concat([dfOne, df], axis=0)

            # Calculate accuracy (or another metric)
            accuracy = grid_search.best_score_

            # Save the best model based on accuracy
            if accuracy > self.best_accuracy:
                self.best_accuracy = accuracy
                self.best_model = best_model_instance
                os.makedirs('artifacts', exist_ok=True)
                with open('artifacts/'+self.save_path, 'wb') as f:
                    pickle.dump(best_model_instance, f)
                print(f"Saved new best model: {model_name} with accuracy: {self.best_accuracy:.4f}")

        dfOne.drop_duplicates(inplace=True)
        print(dfOne)
        return dfOne

