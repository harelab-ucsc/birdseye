import pandas as pd
import numpy as np
import sqlite3
import math
import io


class dbConnector:
    def __init__(self, db_name) -> None:
        # Converts np.array to TEXT when inserting
        sqlite3.register_adapter(np.ndarray, self.adapt_array)
        # Converts TEXT to np.array when selecting
        sqlite3.register_converter("array", self.convert_array)
        self.db_c = sqlite3.connect(f'{db_name}.db', detect_types=sqlite3.PARSE_DECLTYPES)
        self.db_c.create_function('sqrt', 1, math.sqrt)
        self.db_c.create_function("pow", 2, self.sqlite_power)


    def boot(self, db_name, sensor):
        self.setupTable(f"{sensor}_poses_{db_name}", "x REAL, y REAL, z REAL, q REAL, u REAL, a REAL, t REAL, rtk_time REAL, alt_time REAL, imu_time REAL")
        self.setupTable(f"{sensor}_images_{db_name}", "save_loc TEXT, rtk_fix INTEGER, time REAL")
        self.setupTable(f"clicks_{db_name}", "x REAL, y REAL")#, health INTEGER")
        self.setupTable(f"parameters_{db_name}", f"sensorID TEXT UNIQUE, resolution array, intrinsics1 array, intrinsics2 array, extrinsics array")
        # self.setupTable(f"relations_{flight_name}", "img_loc TEXT, pred_pixel_x array, pred_pixel_y array, blob_center_x array, blob_center_y array")


    def sqlite_power(self, x, n):
        return int(x)**n


    def adapt_array(self, arr):
        """
        adapts numpy array to non-native SQLite datatype
        http://stackoverflow.com/a/31312102/190597 (SoulNibbler)
        """
        out = io.BytesIO()
        np.save(out, arr)
        out.seek(0)
        return sqlite3.Binary(out.read())


    def convert_array(self, text):
        """
        converts non-native SQLite array datatype from binary to
        """
        out = io.BytesIO(text)
        out.seek(0)
        return np.load(out)


    def checkForTable(self, table_name):
        cur = self.db_c.cursor()
        res = cur.execute(f"SELECT name FROM sqlite_master WHERE type='table' AND name='{table_name}';")
        if len(res.fetchall()) == 0:
            return False
        return True


    def diagnostic(self, table, max=0):
        cur = self.db_c.cursor()
        limit = ';'
        if max != 0:
            limit = f" LIMIT {max};"
        res = cur.execute(f"SELECT * FROM {table}" + limit)
        print(res.fetchall())


    def getFrom(self, what, where, max=0, cond=None):
        cur = self.db_c.cursor()
        limit = ';'
        if max != 0:
            limit = f" LIMIT {max};"
        if cond == None:
            cond = ' '
        query = f"SELECT {what} FROM {where}" + " " + cond + limit
        # print(query)
        res = cur.execute(query)
        return res.fetchall()

    #executes arbitrary sql
    # limit max
    # excutes
    # returns iterable over every row

    def dfToTable(self, data, where, over_write=True):
        proc_d = self.checkForTable(where)
        if proc_d == True and over_write == True:
            cur = self.db_c.cursor()
            cur.execute(f"DROP TABLE {where}")
            data.to_sql(name=where, con=self.db_c)
        if proc_d == False:
            data.to_sql(name=where, con=self.db_c)


    def tableToDF(self, where):
        return pd.read_sql_query(f"SELECT * FROM {where}", self.db_c)


    def setupTable(self, table_name, cols):
        cur = self.db_c.cursor()
        res = cur.execute(f"SELECT name FROM sqlite_master WHERE type='table' AND name='{table_name}';")
        if len(res.fetchall()) == 0:
            cur.execute(f"CREATE TABLE {table_name}({cols})")
            self.db_c.commit()


    def insertInto(self, table_name, cols, vals):
        cur = self.db_c.cursor()
        cur.execute(f"INSERT INTO {table_name}({cols}) VALUES({vals})")
        self.db_c.commit()


    def insertIgnoreInto(self.table_name, cols, vals):
        cur = self.db_c.cursor()
        cur.execute(f"INSERT IGNORE INTO {table_name}({cols}) VALUES({vals})")
        self.db_c.commit()


    def insertClicks(self, table_name, vals):
        cur = self.db_c.cursor()
        # cur.executemany(f"INSERT INTO {table_name} (x, y, health) VALUES(?,?,?)", vals)
        cur.executemany(f"INSERT INTO {table_name} (x, y) VALUES(?,?)", vals)
        self.db_c.commit()


    def updateDataDetections(self, table_name, vals):
        cur = self.db_c.cursor()
        cur.executemany(f"UPDATE {table_name} SET blob_center_x = ?, blob_center_y = ? WHERE img_loc = ?;", vals)
        self.db_c.commit()


    def dropTable(self, table_name):
        cur = self.db_c.cursor()
        # check if table exists
        if self.checkForTable(table_name) == False:
            return
        cur.execute(f"DROP TABLE {table_name}")
        self.db_c.commit()


    def insertMany(self, table_name, cols, vals):
        cur = self.db_c.cursor()
        num_columns = len(cols.split(","))
        # Build the SQL statement to insert rows
        placeholders = ",".join(["?" for _ in range(num_columns)])
        cur.executemany(f"INSERT INTO {table_name}({cols}) VALUES({placeholders})", vals)
        self.db_c.commit()
