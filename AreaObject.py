# !/usr/bin/env python3
# -*- coding: utf-8 -*-
import pymysql
import numpy as np
import pandas as pd

class Area():
    """
    This object has attributes and methods to get data from database.
    """
    def __init__(self, **kwargs):
        self.name = kwargs.get('name')
        self.MSOA = kwargs.get('msoa')

    # Get the number of crime cases in every month with list format
    def SetMSOA(self, msoa):
        self.MSOA = msoa
    def GetFrequency(self):
        """
        This method fetches the monthly crime cases frequency from the database
        """
        sql = f"SELECT crime_date, COUNT(*) AS cases FROM CrimeCasesByMSOA where MSOA_code=\'{self.MSOA}\' group by crime_date order by crime_date;"
        temp_dict = self.MysqlConnection(sql)
        res_df = pd.DataFrame(temp_dict)
        return res_df.loc[:, 'cases'].values.tolist()

    # Get the number of crime cases in every month with array format
    def GetFrequency_array(self):
        """
        This method fetches the monthly crime cases frequency from the database and transfer them into array format
        """
        sql = f"SELECT crime_date, COUNT(*) AS cases FROM CrimeCasesByMSOA where MSOA_code=\'{self.MSOA}\' group by crime_date order by crime_date;"
        temp_dict = self.MysqlConnection(sql)
        res_df = pd.DataFrame(temp_dict)
        return res_df.loc[:, 'cases'].values

    # Get the number of crime cases in the first 6 months as the primary sample
    def GetPrimary(self):
        """
        This method fetches the number of cases in first 6 months as primary set
        """
        sql = f"SELECT crime_date, COUNT(*) AS cases FROM CrimeCasesByMSOA where MSOA_code=\'{self.MSOA}\' and (crime_date < \'2019-3-12\') group by crime_date;"
        temp_dict = self.MysqlConnection(sql)
        res_df = pd.DataFrame(temp_dict)
        return res_df['cases'].to_list()

    def GetAuxiliary(self):
        """
        Get the number of crime cases in months before lockdown as auxiliary sample.
        The data during and post lockdown is excluded to avoid the impact from pandemic
        """
        sql = f"SELECT crime_date, COUNT(*) AS cases FROM CrimeCasesByMSOA where MSOA_code=\'{self.MSOA}\' and (crime_date < \'2020-03-12\') group by crime_date;"
        temp_dict = self.MysqlConnection(sql)
        res_df = pd.DataFrame(temp_dict)
        return res_df['cases'].to_list()

    # Get the number of crime cases during the period from '2019-6' to '2020-2' as test sample.
    def GetTest(self):
        """
        Get the number of crime cases in months after the 6th months as the TEST set.
        """
        sql = f"SELECT crime_date, COUNT(*) AS cases FROM CrimeCasesByMSOA where MSOA_code=\'{self.MSOA}\' and (crime_date > \'2020-2-12\') and (crime_date < \'2021-4-12\')group by crime_date;"
        temp_dict = self.MysqlConnection(sql)
        res_df = pd.DataFrame(temp_dict)
        return res_df['cases'].to_list()

    def GetLocationInfo(self):
        """
        Get geographical location, longitude and latitude of the given area code.
        Variables:
        * tempRes: It restores the dict format data fetched from database.
        """
        sql = f"SELECT longitude, latitude FROM AreaLocation WHERE MSOA_code=\'{self.MSOA}\';"
        tempRes = self.MysqlConnection(sql)[0]
        return tempRes['longitude'], tempRes['latitude']

    def GetIncome(self):
        """
        Get average income level of the given area code.
        """
        sql = f"SELECT income FROM MSOA where MSOA_code=\'{self.MSOA}\';"
        return self.MysqlConnection(sql)[0]['income']
    def GetPopulation(self):
        """
        Get population information of the given area.
        """
        sql = f"SELECT Population FROM MSOA where MSOA_code=\'{self.MSOA}\';"
        return self.MysqlConnection(sql)[0]['Population']
    def GetInfoVector(self):
        resList = [self.GetLocationInfo()[0], self.GetLocationInfo()[1], self.GetIncome(), self.GetPopulation()]
        resArray = np.array(resList)
        resArray = resArray.reshape(1, resArray.shape[0])
        return resArray
    def MysqlConnection(self, sql):
        """
        Initialize the connection to the database server.
        """
        connection = pymysql.connect(host='localhost',
                                     user='root',
                                     password='DSaiYT1314',
                                     database='crime',
                                     cursorclass=pymysql.cursors.DictCursor)
        with connection.cursor() as cursor:
            cursor.execute(sql)
            result = cursor.fetchall()
        connection.commit()
        return result
if __name__ == "__main__":
    a = Area(MSOA="E02001343")
    b = Area(MSOA="E02001434")
    print(a.MSOA)








