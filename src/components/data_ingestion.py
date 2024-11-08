#data ingestion is accessing and transporting data from a source or sources to a target
#The data is sanitized and transformed into a uniform format by using an extract, transform, 
#load (ETL) process or extract load transform process (ELT), to manage the data lifecycle effectively.
import os
import sys
from src.exception import CustomException
from src.logger import logging
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass

from src.components.data_transformation import DataTransformationConfig
from src.components.data_transformation import DataTransformation

from src.components.model_trainer import ModelTrainerConfig
from src.components.model_trainer import ModelTrainer

#when you just have to add variables in the class you can use dataclass decorator
'''
 it automatically generates special methods for the class, 
 such as __init__(), __repr__(), __eq__(), and others, based on the 
 class attributes you define. When you decorate a class with @dataclass, 
 Python takes the class attributes you define and uses them to generate 
 an initializer (__init__) and other useful methods automatically.

 @dataclass
 class Student:
    name: str
    age: int
    grade: float = 0.0  # default value

__init__: Initializes instance variables with the types and default values (if provided).
__repr__: Returns a string representation of the object for debugging purposes.
__eq__: Compares two instances for equality based on their attribute values.
'''
@dataclass
class DataIngestionConfig:
    train_data_path: str=os.path.join('artifacts','train.csv')
    test_data_path: str=os.path.join('artifacts','test.csv')
    raw_data_path: str=os.path.join('artifacts','data.csv')

class DataIngestion:
    def __init__(self):
        self.ingestion_config=DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Entered the data ingestion method or component")
        try:
            df=pd.read_csv('notebook/data/diabetes.csv')
            logging.info('Exported the dataset')

            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)

            df.to_csv(self.ingestion_config.raw_data_path, index=False,header=True)

            logging.info('Train test split initiated')
            train_set, test_set=train_test_split(df, test_size=0.2, random_state=42)

            train_set.to_csv(self.ingestion_config.train_data_path, index=False,header=True)
             
            test_set.to_csv(self.ingestion_config.test_data_path, index=False,header=True)

            logging.info("Ingestion of the data is completed")

            return(
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )
        except Exception as e:
            raise CustomException(e, sys) 
        
if __name__=="__main__":
    obj=DataIngestion()
    train_data,test_data=obj.initiate_data_ingestion()

    data_transformation=DataTransformation()
    train_arr, test_arr,_= data_transformation.initiate_data_transformation(train_data,test_data)

    modeltrainer=ModelTrainer()
    print("Best Model:", modeltrainer.initiate_model_trainer(train_arr, test_arr))


