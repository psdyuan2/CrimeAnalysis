import pandas as pd
from AreaObject import Area
import pymysql
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from matplotlib import pyplot as plt


class AreaUK(Area):
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

    def getFeature(self, msoa_code):
        sql = f'select * from msoa_uk2 where msoa_code={msoa_code};'
        return self.MysqlConnection(sql)


class Dataset():
    def uk_msoa_info(self, fraction):
        total_row_sql = Area().MysqlConnection(f"select count(*) as total_row_count from msoa_uk2;")
        total_row_sql = total_row_sql[0].get("total_row_count")
        total_row = round(total_row_sql * fraction)
        fetch_sql = f"select * from msoa_uk2 order by rand() limit {total_row}"
        d = pd.DataFrame(Area().MysqlConnection(fetch_sql))
        d.set_index("msoa_code", inplace=True)
        d.dropna(inplace=True)
        d["population"] = d["population"].str.replace(",","")
        d["population"] = d["population"].str.replace(" ", "")
        d["annual_income"] = d["annual_income"].str.replace(",", "")
        print(f"d is :{d}")
        return d


class Cluster():
    # Using standard scaler, mean and std to make scalar processing
    def data_prepeparing(self, data):
        scalar = StandardScaler()
        scalar.fit(data)
        scalared = scalar.transform(data)
        return scalared


if __name__ == "__main__":
    d = Dataset().uk_msoa_info(0.01)
    d = d.drop(['msoa_name',"annual_income","population"], axis=1)
    d = Cluster().data_prepeparing(d)
    #print(Cluster().data_prepeparing(d))
    res = KMeans(8, random_state=1).fit_predict(d)
    plt.scatter(d[:, 0], d[:, 1], c=res)
    print(f"res is {res}")
    plt.show()