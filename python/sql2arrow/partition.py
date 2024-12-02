from abc import ABC, abstractmethod
class PartitionFuncSpec(ABC):

    @abstractmethod
    def name(self):
        pass

    @abstractmethod
    def is_valid(self) -> bool:
        return False

class IcebergPartitionFuncSpec(PartitionFuncSpec):
    def __init__(self):
        self._partitions = {}
    
    def name(self):
        return "iceberg"
    
    def add_partition(self, column_name, transform_str):
        if not isinstance(column_name, str) or not column_name:
            raise ValueError("column_name is not str or empty")

        if not isinstance(transform_str, str) or not transform_str:
            raise ValueError("transform_str is not str or empty")
        
        self._partitions[column_name] = transform_str

    def partitions(self):
        return self._partitions
    
    def is_valid(self):
        if len(self._partitions) == 0:
            return False
        return True

class IcebergTransforms:
    def bucket(num : int):
        if num <= 0:
            raise ValueError(f"invalid num {num} for iceberg bucket transform")
        return f"bucket[{num}]"
    def truncate(width : int):
        if width <= 0:
            raise ValueError(f"invalid width {width} for iceberg truncate transform")
        return f"truncate[{width}]"
    def identity():
        return "identity"
    def year():
        return "year"
    def month():
        return "month"
    def day():
        return "day"
    def hour():
        return "hour"