import datetime
import os
import time

import pymysql

HOST = "127.0.0.1"
USERNAME = "root"
PASSWORD = "123456"
DATABASE = "md_admin"


class DB:
    connection = None
    cursor = None

    def __init__(self):
        self.connection = pymysql.connect(
            host=HOST,
            database=DATABASE,
            user=USERNAME,
            password=PASSWORD
        )

        try:
            self.cursor = self.connection.cursor()
            self.cursor.execute("select @@version ")
            version = self.cursor.fetchone()
            if version:
                print('Running version: ', version)
            else:
                print('Not connected.')
        except Exception as e:
            print("Error while connecting to MySQL", e)

    def close(self):
        self.connection.close()

    def insertDetectionLog(self, img_path, location):
        sql = "INSERT INTO `detection_logs` (`img_path`, `location`, `created_at`) VALUES (%s, %s, %s)"
        created_at = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.cursor.execute(sql, (img_path, location, created_at))
        self.connection.commit()


if __name__ == '__main__':
    db = DB()
    db.insertDetectionLog("test", "test")
    db.close()
