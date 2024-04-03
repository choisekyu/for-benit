import pyodbc

from cr_utils import check_time


class CRDB(object):
    def __init__(self,
                 server: str,
                 database: str,
                 username: str,
                 password: str,
                 table: str,
    ) -> None:
        self._conn_str = f"""Driver={{ODBC Driver 18 for SQL Server}};Server={
            server};Database={database};Uid={username};Pwd={
            password};TrustServerCertificate=Yes"""
        self.columns = 'EQUIPMENT_ID, PARAMETER_ID, VALUE, TXN_TIME'
        self.table = table

    @check_time
    def query(self, start: str, finish: str, table: str = None) -> tuple:
        table = table or self.table
        qstr = f"""SELECT {self.columns} FROM {table} WITH(NOLOCK) WHERE 1=1 \
        AND EQUIPMENT_ID IN ('RM01_M','RM01_W','RM05_M','RM05_W') \
        AND TXN_TIME>='{start}' AND TXN_TIME<'{finish}'"""
        with pyodbc.connect(self._conn_str) as conn:
            cursor = conn.cursor()
            cursor.execute(qstr)
            result = cursor.fetchall()

        return result
