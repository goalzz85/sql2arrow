use std::{collections::HashMap, hash::Hash};
use arrow::{array::BufferBuilder, buffer::ScalarBuffer, datatypes::Int8Type};
use arrow_array::{ArrayRef, PrimitiveArray};
use arrow_schema::{DataType, TimeUnit};
use iceberg::{spec::Transform, transform::create_transform_function};
use pyo3::prelude::*;
use crate::types::ColumnArrStrDef;
use anyhow::anyhow;
use std::sync::Arc;



#[derive(Debug, PartialEq, Eq, Clone, Hash)]
pub(crate) enum PartitionType {
    Default,
    Iceberg,
}

impl std::fmt::Display for PartitionType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            PartitionType::Default => write!(f, "Default"),
            PartitionType::Iceberg => write!(f, "Iceberg"),
        }
    }
}

pub trait PartitionFunc: Send + Sync {
    fn partition_type(&self) -> PartitionType;
    fn transform(&self, arr_refs : &Vec<ArrayRef>) -> anyhow::Result<Vec<ArrayRef>>;
}

pub type PartitionKey = Vec<u8>;


pub fn py_partition_func_spec_obj_to_rust(partition_func_spec : &PyObject, columns_def : &ColumnArrStrDef) -> anyhow::Result<Arc<dyn PartitionFunc>> {
    Python::with_gil(|py| -> anyhow::Result<Arc<dyn PartitionFunc>> {
        let thismod = py.import_bound("sql2arrow")?;
        let partition_func_spec_py_class = thismod.getattr("partition")?.getattr("PartitionFuncSpec")?;
        if !partition_func_spec.bind(py).is_instance(&partition_func_spec_py_class)? {
            return Err(anyhow!("Invalid PartitionFuncSpec Object!"));
        }

        let partition_type_name : String = partition_func_spec.bind(py).call_method0("name")?.extract()?;

        match partition_type_name.as_str() {
            "iceberg" => IceBergPartition::from_pyobj(py, &partition_func_spec, columns_def),
            _ => {
                return Err(anyhow!("Not supported partition func spec"));
            }
        }
    })
}

pub struct DefaultPartition {

}
impl PartitionFunc for DefaultPartition {
    fn partition_type(&self) -> PartitionType {
        PartitionType::Default
    }

    fn transform(&self, arr_refs : &Vec<ArrayRef>) -> anyhow::Result<Vec<ArrayRef>> {
        let arr_len = arr_refs[0].len();
        let mut buf_builder = BufferBuilder::<i8>::new(arr_len);
        buf_builder.append_n_zeroed(arr_len);
        let buf = buf_builder.finish();
        let arr = PrimitiveArray::<Int8Type>::new(ScalarBuffer::new(buf, 0, arr_len), None);

        Ok(vec![Arc::new(arr)])
    }
}


pub struct IceBergPartition {
    col_idxs : Vec<usize>,
    transforms: Vec<Transform>
}

impl IceBergPartition {

    fn from_pyobj(py : Python<'_>, partition_func_spec : &PyObject, columns_def : &ColumnArrStrDef) -> anyhow::Result<Arc<dyn PartitionFunc>> {
        let partitions_dict :HashMap<String, String> = partition_func_spec.bind(py).call_method0("partitions")?.extract()?;
        let mut col_partition_defs : Vec<(&str, &str)> = Vec::with_capacity(partitions_dict.len());
        for (col, transform) in partitions_dict.iter() {
            col_partition_defs.push((col.as_ref(), transform.as_ref()));
        }

        return Ok(Arc::new(Self::from(&col_partition_defs, columns_def)?));
    }

    /**
     * col_partition_defs:
     * [
     *     ("column name", "parition transform string")
     * ]
     */
    fn from(col_partition_defs : &Vec<(&str, &str)>, columns_def : &ColumnArrStrDef) -> anyhow::Result<Self> {
        if col_partition_defs.is_empty() {
            return Err(anyhow!("partition transforms is empty"));
        }

        let mut col_idxs = Vec::<usize>::with_capacity(col_partition_defs.len());
        let mut col_transforms = Vec::<Transform>::with_capacity(col_partition_defs.len());

        for (col_name, transform_str) in col_partition_defs {
            let mut is_have_col = false;
            for (idx, (col_name_def, _col_data_type_def)) in columns_def.iter().enumerate() {
                if col_name.eq(col_name_def) {
                    is_have_col = true;
                    col_idxs.push(idx);
                }
            }
            if !is_have_col {
                return Err(anyhow!("not found column name {:?}", col_name));
            }

            //get iceberg transform
            let tf = transform_str.parse()?;
            col_transforms.push(tf);
        }

        return Ok(IceBergPartition{
            col_idxs: col_idxs,
            transforms: col_transforms,
        });
    }
}


impl PartitionFunc for IceBergPartition {
    fn partition_type(&self) -> PartitionType {
        PartitionType::Iceberg
    }

    fn transform(&self, arr_refs : &Vec<ArrayRef>) -> anyhow::Result<Vec<ArrayRef>> {
        let mut res_arr_refs = Vec::<ArrayRef>::with_capacity(arr_refs.len());

        for (i, col_idx) in self.col_idxs.iter().enumerate() {
            if let Some(tf) = self.transforms.get(i) {
                let func = create_transform_function(tf)?;
                let arr_ref= arr_refs.get(col_idx.clone()).unwrap();
                let res_arr_ref = func.transform(arr_ref.to_owned())?;
                res_arr_refs.push(res_arr_ref);
            } else {
                return Err(anyhow!("not found transform for col idx {:?}", col_idx));
            }
        }
        
        return Ok(res_arr_refs);
    }
}


pub fn get_parition_key_from_first_val(partition_val_arr_refs: &Vec<ArrayRef>) -> anyhow::Result<PartitionKey> {
    let mut pk = Vec::<u8>::new();

    for arr_ref in partition_val_arr_refs {
        if arr_ref.is_empty() {
            return Err(anyhow!("get partition key with empty partition_val_arr_refs"));
        }

        match arr_ref.data_type() {
            DataType::Int8 => {
                let v = arr_ref.as_any().downcast_ref::<arrow::array::Int8Array>().unwrap().value(0);
                pk.extend_from_slice(&v.to_be_bytes());
            }
            DataType::Int32 => {
                let v = arr_ref.as_any().downcast_ref::<arrow::array::Int32Array>().unwrap().value(0);
                pk.extend_from_slice(&v.to_be_bytes());
            },
            DataType::Int64 => {
                let v = arr_ref.as_any().downcast_ref::<arrow::array::Int64Array>().unwrap().value(0);
                pk.extend_from_slice(&v.to_be_bytes());
            },
            DataType::Decimal128(_, _) => {
                let v = arr_ref.as_any().downcast_ref::<arrow::array::Decimal128Array>().unwrap().value(0);
                pk.extend_from_slice(&v.to_be_bytes());
            },
            DataType::Date32 => {
                let v = arr_ref.as_any().downcast_ref::<arrow::array::Date32Array>().unwrap().value(0);
                pk.extend_from_slice(&v.to_be_bytes());
            },
            DataType::Time64(TimeUnit::Microsecond) => {
                let v = arr_ref.as_any().downcast_ref::<arrow::array::Time64MicrosecondArray>().unwrap().value(0);
                pk.extend_from_slice(&v.to_be_bytes());
            },
            DataType::Timestamp(TimeUnit::Microsecond, _) => {
                let v = arr_ref.as_any().downcast_ref::<arrow::array::TimestampMicrosecondArray>().unwrap().value(0);
                pk.extend_from_slice(&v.to_be_bytes());
            },
            _ => {
                return Err(anyhow!("not support partition value data type for creating partition key"));
            }
        }
    }
    
    return Ok(pk);
}