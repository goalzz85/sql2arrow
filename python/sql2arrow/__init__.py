
from typing import List
from enum import Enum
from . import sql2arrow as s2a
from .sql2arrow import enable_log
from . import partition

class Dialect(Enum):
    DEFAULT = None
    MYSQL = "mysql"
    PGSQL = "postgresql"

class CompressionType(Enum):
    NONE = None
    GZIP = "gzip"
    SNAPPY = "snappy"

class ArrowTypes:
    def int8():
        return "Int8"
    def int16():
        return "Int16"
    def int32():
        return "Int32"
    def int64():
        return "Int64"
    
    def uint8():
        return "UInt8"
    def uint16():
        return "UInt16"
    def uint32():
        return "UInt32"
    def uint64():
        return "UInt64"

    def float32():
        return "Float32"
    def float64():
        return "Float64"

    def boolen():
        return "Boolean"
    
    def utf8():
        return "Utf8"
    
    def decimal128(precision, scale):
        return f"Decimal128({precision},{scale})"
    
    def decimal256(precision, scale):
        return f"Decimal256({precision},{scale})"
    
    


class Column:
    def __init__(self, name, column_arrow_type : str):
        self.name = name
        self.type = column_arrow_type

def parse_sql(sql : str, columns : List[Column], dialect : Dialect = Dialect.MYSQL):
    sql_data = sql.encode()
    column_defs = [(c.name, c.type) for c in columns]
    datas = s2a.load_sqls_with_dataset([sql_data], column_defs, None, None, dialect.value)
    return datas[0]

class SQLFile2ArrowIter:
    def __init__(
            self,
            sqlfile_paths : List[str],
            columns : List[Column],
            thread_num : int,
            batch_data_threshold : int = 0,
            compression : CompressionType = CompressionType.NONE,
            dialect : Dialect = Dialect.MYSQL,
            partition_func = None
            ):
        column_defs = [(c.name, c.type) for c in columns]
        self.inner_loader = s2a.SQLFile2ArrowLoader(
            sqlfile_paths,
            column_defs,
            thread_num,
            batch_data_threshold,
            compression and compression.value,
            dialect and dialect.value,
            partition_func
        )

    def __iter__(self):
        return self
    
    def __next__(self):
        data = self.inner_loader.next_batch_data()
        if data is None:
            raise StopIteration
        else:
            return data