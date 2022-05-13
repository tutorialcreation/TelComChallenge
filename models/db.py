import pandas as pd
from sqlalchemy import types, create_engine
from sqlite3 import Error
import csv
import sys, os
sys.path.append(os.path.abspath(os.path.join('..')))

class DBOps:
    """
    What this script does:
    - inserts data from csv into sqlite (online)
    """


    def __init__(self,is_online=True):
        if is_online:
            try:
                self.conn = create_engine('sqlite:///satisfaction.db') # ensure this is the correct path for the sqlite file. 
                print("SQLITE Connection Sucessfull!!!!!!!!!!!")
            except Exception as err:
                print("SQLITE Connection Failed !!!!!!!!!!!")
                print(err)
        


    def get_engine(self):
        """
        - this function simply returns the connection
        """
        return self.conn

    def get_df(self):
        """
        - this function returns the data
        to be inserted into the sql table
        """
        df = pd.read_csv("data/satisfaction.csv")
        return df

    
    def execute_from_script(self,sql_script):
        """
        - this function executes commands
        that come streaming in from sql_scripts
        """
        try:
            sql_file = open(sql_script)
            sql_ = sql_file.read()
            sql_file.close()


            sql_commands = sql_.split(";")
            for command in sql_commands:
                if command:
                    self.conn.execute(command)
            print("Successfully created table")
        except Error as e:
            print(e)
        return
    

    def insert_update_data(self,table):
        """
        - this function pushes data into the table
        """
        df = self.get_df()
        df.to_sql(table, con=self.conn, if_exists='replace')
        print("Successfully pushed the data into the database")
        return 
    
if __name__ == "__main__":
    print("Test DBOpsfile")