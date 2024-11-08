import sys
import logging
from src import logger
#i just imported logger file in exception file so that when i run python src/exception.py
#the log ofdivide by zero exception can be created


def error_message_detail(error, error_detail:sys):
    _,_,exc_tb=error_detail.exc_info()
    file_name=exc_tb.tb_frame.f_code.co_filename
    error_message="Error occured in python script name [{0}] line number [{1}] error essage[{2}]".format( 
        file_name, exc_tb.tb_lineno,str(error))
    return error_message    



class CustomException(Exception):
    def __init__(self, error_message, error_detail:sys):
        super().__init__(error_message)
        self.error_message=error_message_detail(error_message, error_detail=error_detail)

    def __str__(self):
        return self.error_message

if __name__=="__main__":
    try:
        a=1/0
    except Exception as e:
        logging.info("Divide by zero")
        raise CustomException(e, sys)



'''
This is common whenever you try to catch the exception by using try and catch anywhere in the code of this project or meta-data, the same message:

error_message="Error occured in python script name [{0}] line number [{1}] error essage[{2}]".format( 
        file_name, exc_tb.tb_lineno,str(error))
        
will appear according to the error's individual errors
'''