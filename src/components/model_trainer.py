import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from src.exception import CustomException
from src.logger import logging

from src.utils import save_object
from src.utils import evaluate_model
from dataclasses import dataclass
import os,sys

@dataclass
class ModelTrainerconfig:
    trained_model_path: str = os.path.join("artifacts", "model.pkl") # Path to save the trained model


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerconfig()

    def initiate_model_traininig(self,train_array,test_array):
        try:
            logging.info('Splitting dependent and indepdent the data into train and test')
            X_train,y_train,X_test,y_test=(
                train_array[:,:-1],train_array[:,-1],
                test_array[:,:-1],test_array[:,-1]
            )

            models={
                'LinearRegression':LinearRegression(),
                'Ridge':Ridge(),
                'Lasso':Lasso(),
                'ElasticNet':ElasticNet(),
                'RandomForestRegressor':RandomForestRegressor(),
                'DecisionTreeRegressor':DecisionTreeRegressor(),
                'KNeighborsRegressor':KNeighborsRegressor()
            }

            model_report:dict=evaluate_model(X_train,y_train,X_test,y_test,models)
            print('\n====================================================================================')
            logging.info(f'Model Report: {model_report}')
            
            # to get best model score from dictionary
            best_model_score=max(sorted(model_report.values()))

            best_model_name=list(model_report.keys())[list(model_report.values()).index(best_model_score)]

            best_model = models[best_model_name]

            print(f'Best Model Found, Model Name: {best_model_name}, Model Score: {best_model_score}')
            print('\n====================================================================================')
            logging.info(f'Best Model Found, Model Name{best_model_name}, Model Score: {best_model_score}')

            save_object(file_path=self.model_trainer_config.trained_model_path,obj=best_model)

        except Exception as e:
            raise CustomException(e,sys)
