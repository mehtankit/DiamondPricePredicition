from sklearn.impute import SimpleImputer ## Handling Missing Values
from sklearn.preprocessing import StandardScaler # Handling Feature Scaling
from sklearn.preprocessing import OrdinalEncoder # Ordinal Encoding
## pipelines
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import pandas as pd
import numpy as np

from src.exception import CustomException
from src.logger import logging
import sys,os
from dataclasses import dataclass

from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path=os.path.join("artifacts","preprocessor.pkl")

class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()

    def get_data_transformation_object(self):
        try:
            logging.info("Data Transformation Started")

            ## categorical and numerical columns
            categorical_columns = ['cut','color','clarity']
            numerical_columns = ['carat','depth','table','x','y','z']

            ## define the custom ranking for each ordinal column
            cut_categories = ['Fair', 'Good', 'Very Good','Premium','Ideal']
            color_categories = ['D', 'E', 'F', 'G', 'H', 'I', 'J']
            clarity_categories = ['I1','SI2','SI1','VS2','VS1','VVS2','VVS1','IF']

            logging.info("Data Transformation Pipeline Started")  

            #numerical pipeline
            num_pipeline = Pipeline([
                ('imputer', SimpleImputer(strategy='mean')),
                ('scaler', StandardScaler())
            ])

            #categorical pipeline
            cat_pipeline = Pipeline([
                ('ordinal_encoder', OrdinalEncoder(categories=[cut_categories,color_categories,clarity_categories]))
            ])

            #full pipeline
            preprocessor = ColumnTransformer([
                ('num_pipeline', num_pipeline, numerical_columns),
                ('cat_pipeline', cat_pipeline, categorical_columns)
            ])

            logging.info("Data Transformation Pipeline Completed")

            return preprocessor

        except Exception as e:
            logging.error("Exception occured in Data Transformation")
            raise CustomException(e,sys)
    
    def initiate_data_transformation(self,train_data_path,test_data_path):

        try:
            ## read the data
            train_df=pd.read_csv(train_data_path)
            test_df=pd.read_csv(test_data_path)

            logging.info("Read train and test data completed")
            logging.info(f"Train Dataframe Head: \n{train_df.head().to_string()}")
            logging.info(f"Test Dataframe Head: \n{test_df.head().to_string()}")

            logging.info('Obtaining preprocessing object')

            preprocessing_obj=self.get_data_transformation_object()

            target_column='price'
            drop_columns=[target_column,'id']

            ## dividing the dataset into indepdent and dependent features
            ## training data
            input_feature_train_df=train_df.drop(drop_columns,axis=1)
            target_feature_train_df=train_df[target_column]

            ## testing data
            input_feature_test_df=test_df.drop(drop_columns,axis=1)
            target_feature_test_df=test_df[target_column]

            ## data transformation
            input_feature_train_arr=preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr=preprocessing_obj.transform(input_feature_test_df)

            train_arr=np.c_[input_feature_train_arr,np.array(target_feature_train_df)]
            test_arr=np.c_[input_feature_test_arr,np.array(target_feature_test_df)]

            save_object(
                self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )

            logging.info("Preprocssing Completed")

            return(
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path
            )


        except Exception as e:
            logging.error("Exception occured in Data Transformation")
            raise CustomException(e,sys)