
from typing import List
from enum import Enum
from . import sql2arrow as s2a
from .sql2arrow import enable_log
from . import partition

class Dialect(Enum):
    DEFAULT = None
    MYSQL = "mysql"

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
    


class Column:
    def __init__(self, name, column_arrow_type : str):
        self.name = name
        self.type = column_arrow_type

def parse_sql(sql : str, columns : List[Column], dialect : Dialect = Dialect.MYSQL):
    sql_data = sql.encode()
    datas = _load_sqls_with_sql_datas([sql_data], columns, None, None, dialect.value)
    return datas[0]

def load_sqls(sql_paths : List[str], columns : List[Column], compression : CompressionType = CompressionType.NONE, dialect : Dialect = Dialect.MYSQL, max_thread_num = 32, pyarrow_fs = None):
    if len(sql_paths) == 0:
        return []
    if len(sql_paths) > max_thread_num:
        raise ValueError(f"Input SQL file num {len(sql_paths)} exceeds the maximum thread num {max_thread_num}.")

    if pyarrow_fs is not None:
        sql_datas = _load_data_file_by_pyarrow_fs(sql_paths, pyarrow_fs)
        return _load_sqls_with_sql_datas(sql_datas, columns, None, compression and compression.value, dialect and dialect.value)

    return _load_sqls(sql_paths, columns, None, compression and compression.value, dialect and dialect.value)

def load_sqls_with_partition_func(sql_paths : List[str], columns : List[Column], partition_func = None, compression : CompressionType = CompressionType.NONE,  dialect : Dialect = Dialect.MYSQL, max_thread_num = 32, pyarrow_fs = None):
    if len(sql_paths) == 0:
        return []
    if len(sql_paths) > max_thread_num:
        raise ValueError(f"Input SQL file num {len(sql_paths)} exceeds the maximum thread num {max_thread_num}.")

    if pyarrow_fs is not None:
        sql_datas = _load_data_file_by_pyarrow_fs(sql_paths, pyarrow_fs)
        return _load_sqls_with_sql_datas(sql_datas, columns, partition_func, compression and compression.value, dialect and dialect.value)

    return _load_sqls(sql_paths, columns, partition_func, compression and compression.value, dialect and dialect.value)

def _load_sqls(sql_paths : List[str], columns : List[Column], partition_func = None, compression = None, dialect = None):
    column_defs = [(c.name, c.type) for c in columns]
    return s2a.load_sqls(sql_paths, column_defs, partition_func, compression, dialect)

def _load_sqls_with_sql_datas(sql_dataset : List[bytes], columns : List[Column], partition_func = None, compression = None, dialect = None):
    column_defs = [(c.name, c.type) for c in columns]
    return s2a.load_sqls_with_dataset(sql_dataset, column_defs, partition_func, compression, dialect)


def _load_data_file_by_pyarrow_fs(sql_paths, pyarrow_fs):
    from pyarrow.fs import FileSystem
    from concurrent.futures import ThreadPoolExecutor, as_completed
    
    if not issubclass(pyarrow_fs.__class__, FileSystem):
        raise ValueError(f"pyarrow_fs is not a subclass of pyarrow.fs.FileSystem.")

    def read_file_to_data(sql_path, pyarrow_fs):
        with pyarrow_fs.open_input_file(sql_path) as f:
            return f.readall()
    
    all_datas = [None] * len(sql_paths)

    with ThreadPoolExecutor(max_workers=5) as executor:
        # 提交任务
        future_to_data = {executor.submit(read_file_to_data, sql_path, pyarrow_fs): i for i, sql_path in enumerate(sql_paths)}
        for future in as_completed(future_to_data):
            i = future_to_data[future]
            data = future.result()  # 获取任务结果
            all_datas[i] = data
    return all_datas