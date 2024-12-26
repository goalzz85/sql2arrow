# SQL2Arrow

This is a Python library that provides convenient and high-performance methods to parse INSERT SQL statements into Arrow arrays. It's very useful for analyzing data dumped by mysqldump or other tools.

## How to use

### Installation

Install the latest SQL2arrow version with:

```bash
pip install sql2arrow
```

### Parsing SQL str
```python
import sql2arrow

sql_str = '''
INSERT INTO `region` VALUES
	('', '', '2023-01-31 18:00:48', '2023-01-31 18:00:48', ''),
	('1541947646568607746', 'region name', '2022-06-29 08:52:21', '2022-06-29 08:52:21', 'D99'),
	('1541947680890597378', 'region name1', '2022-06-29 08:52:29', '2022-06-29 08:52:29', 'D98'),
	('620422117205', 'region name7', '2021-10-25 18:23:48', '2021-10-25 18:23:48', 'D620422117');
'''

columns = [
    sql2arrow.Column("region_code", sql2arrow.ArrowTypes.utf8()),
    sql2arrow.Column("region_name", sql2arrow.ArrowTypes.utf8()),
    sql2arrow.Column("create_time", sql2arrow.ArrowTypes.utf8()),
    sql2arrow.Column("update_time", sql2arrow.ArrowTypes.utf8()),
    sql2arrow.Column("parent_region_code", sql2arrow.ArrowTypes.utf8())
]

arrow_data = sql2arrow.parse_sql(sql_str, columns)
```


### Parsing sql files

```python
import sql2arrow

sql_paths = [
    "region.sql_0.gz", "region.sql_1.gz","region.sql_2.gz","region.sql_3.gz","region.sql_4.gz","region.sql_5.gz","region.sql_6.gz"
]

columns = [
    sql2arrow.Column("region_code", sql2arrow.ArrowTypes.utf8()),
    sql2arrow.Column("region_name", sql2arrow.ArrowTypes.utf8()),
    sql2arrow.Column("create_time", sql2arrow.ArrowTypes.utf8()),
    sql2arrow.Column("update_time", sql2arrow.ArrowTypes.utf8()),
    sql2arrow.Column("parent_region_code", sql2arrow.ArrowTypes.utf8())
]


partition_func_spec = sql2arrow.partition.IcebergPartitionFuncSpec()
partition_func_spec.add_partition("region_code", sql2arrow.partition.IcebergTransforms.bucket(30))


it = sql2arrow.SQLFile2ArrowIter(
    sql_paths,
    columns,
    4,
    1000,
    sql2arrow.CompressionType.SNAPPY,
    sql2arrow.Dialect.MYSQL,
    partition_func_spec
)

for arr in it:
    print(arr)
```


## arro3

SQL2Arrow uses arro3 as the default Python library for Apache Arrow. Thanks to the [Arrow PyCapsule Interface](https://arrow.apache.org/docs/format/CDataInterface/PyCapsuleInterface.html), we can seamlessly pass Arro3's Array data to other libraries compatible with the Arrow PyCapsule Interface, including PyArrow, Polars (v1.2+), Pandas (v2.2+), NanoArrow, and more, all with zero-copy memory.

```python
# some codes from above

import pyarrow as pa
tables = [pa.Table.from_arrays(a, names=names) for a in arrs]
```
## Limitations

### Dialect
    It currently supports only MySQL and PostgreSQL INSERT statements.