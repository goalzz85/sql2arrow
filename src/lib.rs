mod types;
mod arraybuilder;
mod partition;
mod pylog;
mod loader;


use std::cell::RefCell;
use std::collections::HashMap;
use std::io::Read;
use std::sync::{Arc, Mutex};
use std::{thread, time, vec};
use std::time::Instant;

use anyhow::anyhow;
use arrow::array::{Array, ArrayRef as ArrowArrayRef};
use arrow::compute::{SortColumn, TakeOptions};
use arrow_array::UInt32Array;
use flate2::read::GzDecoder;
use loader::ArrowLoader;
use mimalloc::MiMalloc;
use partition::{get_parition_key_from_first_val, py_partition_func_spec_obj_to_rust, DefaultPartition, PartitionFunc, PartitionKey};
use pyo3::prelude::*;
use pyo3_arrow::error::PyArrowResult;
use pyo3_arrow::PyArray;
use sqlparser::dialect::{self, Dialect};
use sqlparser::ast::{Insert, SetExpr, Statement, Values};
use types::ColumnArrStrDef;


#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

/// A Python module implemented in Rust.
#[pymodule]
fn sql2arrow(m: &Bound<'_, PyModule>) -> PyResult<()> {

    m.add_function(wrap_pyfunction!(load_sqls, m)?)?;
    m.add_function(wrap_pyfunction!(load_sqls_with_dataset, m)?)?;
    m.add_function(wrap_pyfunction!(enable_log, m)?)?;
    m.add_class::<SQLFile2ArrowLoader>()?;
    Ok(())
}

#[pyfunction]
fn enable_log(level:i32) -> anyhow::Result<()> {
    let filter = match level {
        //logging.CRITICAL and logging.ERROR
        50 | 40 => log::LevelFilter::Error,
        //logging.WARNING
        30 => log::LevelFilter::Warn,
        //logging.INFO
        20 => log::LevelFilter::Info,
        //logging.DEBUG
        10 => log::LevelFilter::Debug,
        //logging.NOTSET
        0 => log::LevelFilter::Off,
        _ => {
            return Err(anyhow!("not support log level code: {}", level));
        }
    };
    pylog::enable_log(filter)
}

#[pyfunction]
#[pyo3(signature = (sql_paths, columns, partition_func_spec_obj=None, compression_type=None, dialect=None))]
fn load_sqls(py : Python<'_>, sql_paths : Vec<String>, columns: ColumnArrStrDef, partition_func_spec_obj : Option<PyObject>, compression_type : Option<String>, dialect : Option<String>) -> anyhow::Result<Vec<Vec<PyArray>>> {
    if sql_paths.is_empty() {
        return Err(anyhow!("sql_paths is empty"));
    }

    let mut sql_files = Vec::<SqlFileWrapper>::with_capacity(sql_paths.len());

    for sql_path in &sql_paths {
        let sql_file = std::fs::File::open(sql_path)?;
        sql_files.push(SqlFileWrapper(sql_file));
    }
    
    inner_load_sqls_with_dataset(py, sql_files, columns, partition_func_spec_obj, compression_type, dialect)
}

#[pyfunction]
#[pyo3(signature = (sql_dataset, columns, partition_func_spec_obj=None, compression_type=None, dialect=None))]
fn load_sqls_with_dataset(py : Python<'_>, sql_dataset : Vec<Vec<u8>>, columns: ColumnArrStrDef, partition_func_spec_obj : Option<PyObject>, compression_type : Option<String>, dialect : Option<String>) -> anyhow::Result<Vec<Vec<PyArray>>> {
    inner_load_sqls_with_dataset(py, sql_dataset, columns, partition_func_spec_obj, compression_type, dialect)
}

fn inner_load_sqls_with_dataset<T>(py : Python<'_>, sql_dataset : Vec<T>, columns: ColumnArrStrDef, partition_func_spec_obj : Option<PyObject>, compression_type : Option<String>, dialect : Option<String>) -> anyhow::Result<Vec<Vec<PyArray>>>
where T : Into<Vec<u8>> + Send + 'static
{
    if sql_dataset.is_empty() {
        return Err(anyhow!("sql_dataset is empty"));
    }

    let sql_dataset_len = sql_dataset.len();
    let mut partition_func : Arc<dyn PartitionFunc> = Arc::new(DefaultPartition{});
    let mut is_have_partition_func = false;

    if let Some(py_partition_func_spec_obj) = partition_func_spec_obj {
        partition_func = py_partition_func_spec_obj_to_rust(&py_partition_func_spec_obj, &columns)?;
        is_have_partition_func = true;
    }

    py.allow_threads(|| -> anyhow::Result<Vec<Vec<PyArray>>> {
        let load_start_time = Instant::now();
        let data = if !is_have_partition_func {
            pyinfo!("Starting to load {} sql datasets to Arrow without partition func.", sql_dataset_len);
            match load_without_partition_func(sql_dataset, columns, compression_type, dialect) {
                Ok(pyarrs) => PyArrowResult::Ok(pyarrs),
                Err(e) => Err(pyo3_arrow::error::PyArrowError::PyErr(
                        pyo3::exceptions::PyRuntimeError::new_err(e.to_string())
                    )),
            }
        } else {
            pyinfo!("Starting to load {} sql datasets to Arrow with partition func {}.", sql_dataset_len, partition_func.partition_type());
            match load_with_partition_func(sql_dataset, columns, Some(partition_func), compression_type, dialect) {
                Ok(pyarrs) => PyArrowResult::Ok(pyarrs),
                Err(e) => Err(pyo3_arrow::error::PyArrowError::PyErr(
                    pyo3::exceptions::PyRuntimeError::new_err(e.to_string())
                )),
            }
        }?;

        pyinfo!("load {} sql datasets to Arrow has finished in {} seconds.", sql_dataset_len, load_start_time.elapsed().as_secs_f32());
        Ok(data)
    })

}


struct SqlFileWrapper(std::fs::File);

impl Into<Vec<u8>> for SqlFileWrapper {
    
    fn into(mut self) -> Vec<u8> {
        let file_size : usize = self.0.metadata().unwrap().len().try_into().unwrap();
        let mut buf = Vec::<u8>::with_capacity(file_size);
        let read_size = self.0.read_to_end(&mut buf).unwrap();
        if read_size != file_size {
            assert_eq!(file_size, read_size);
        }

        buf
    }
}

fn load_without_partition_func<T: Into<Vec<u8>> + Send + 'static> (sql_dataset : Vec<T>, columns : ColumnArrStrDef, compression_type_op : Option<String>, dialect_op : Option<String>) -> anyhow::Result<Vec<Vec<PyArray>>> {
    load_with_partition_func(sql_dataset, columns, None, compression_type_op, dialect_op)
}

fn load_with_partition_func<T: Into<Vec<u8>> + Send + 'static>(sql_dataset : Vec<T>, columns : ColumnArrStrDef, partition_func_op : Option<Arc<dyn PartitionFunc>>, compression_type_op : Option<String>, dialect_op : Option<String>) -> anyhow::Result<Vec<Vec<PyArray>>> {

    let thread_num = sql_dataset.len();
    let mut loader = crate::loader::ArrowLoader::new(
        sql_dataset,
        columns,
        thread_num,
        0,
        compression_type_op,
        dialect_op,
        partition_func_op
    );
    
    let data = loader.next_batch_data();
    loader.stop();

    match data {
        Ok(res_pyarrs) => match res_pyarrs {
            Some(pyarrs) => Ok(pyarrs),
            None => Ok(vec![]),
        }
        Err(e) => Err(anyhow!(e.to_string())),
    }
}

// #[pyclass]
// struct SQLFile2ArrowIterator {
//     loader : Arc<Mutex<ArrowLoader<SqlFileWrapper>>>
// }

// #[pymethods]
// impl SQLFile2ArrowIterator {

//     #[new]
//     #[pyo3(signature = (sqlfile_paths, columns, thread_num, batch_data_threshold = 0, compression_type=None, dialect=Some("mysql".to_string()), partition_func_spec_obj=None))]
//     pub fn py_new(
//         sqlfile_paths : Vec<String>, 
//         columns: ColumnArrStrDef,
//         thread_num : usize,
//         batch_data_threshold : usize,
//         compression_type : Option<String>,
//         dialect : Option<String>,
//         partition_func_spec_obj : Option<PyObject>,
//     ) -> PyResult<Self> {

//         if sqlfile_paths.is_empty() {
//             return Err(PyErr::from(anyhow!("sql_paths is empty")));
//         }
    
//         let mut sql_files = Vec::<SqlFileWrapper>::with_capacity(sqlfile_paths.len());
    
//         for sqlfile_path in &sqlfile_paths {
//             let sql_file = std::fs::File::open(sqlfile_path)?;
//             sql_files.push(SqlFileWrapper(sql_file));
//         }

//         let mut partition_func : Arc<dyn PartitionFunc> = Arc::new(DefaultPartition{});

//         if let Some(py_partition_func_spec_obj) = partition_func_spec_obj {
//             partition_func = py_partition_func_spec_obj_to_rust(&py_partition_func_spec_obj, &columns)?;
//         }

//         let loader = ArrowLoader::new(sql_files, columns, thread_num, batch_data_threshold, compression_type, dialect, Some(partition_func));

//         return Ok(Self { loader: Arc::new(Mutex::new(loader)) });
//     }

//     fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
//         slf
//     }

//     fn __next__(slf: PyRefMut<'_, Self>) -> Option<anyhow::Result<Vec<Vec<PyArray>>>> {
//         let loader = slf.loader.clone();
//         slf.py().allow_threads(move ||-> Option<anyhow::Result<Vec<Vec<PyArray>>>>{
//             loader.lock().unwrap().next_batch_data()
//         })
//     }
// }


#[pyclass]
struct SQLFile2ArrowLoader {
    loader : Arc<Mutex<ArrowLoader<SqlFileWrapper>>>
}

#[pymethods]
impl SQLFile2ArrowLoader {

    #[new]
    #[pyo3(signature = (sqlfile_paths, columns, thread_num, batch_data_threshold = 0, compression_type=None, dialect=Some("mysql".to_string()), partition_func_spec_obj=None))]
    pub fn py_new(
        sqlfile_paths : Vec<String>, 
        columns: ColumnArrStrDef,
        thread_num : usize,
        batch_data_threshold : usize,
        compression_type : Option<String>,
        dialect : Option<String>,
        partition_func_spec_obj : Option<PyObject>,
    ) -> PyResult<Self> {

        if sqlfile_paths.is_empty() {
            return Err(PyErr::from(anyhow!("sql_paths is empty")));
        }
    
        let mut sql_files = Vec::<SqlFileWrapper>::with_capacity(sqlfile_paths.len());
    
        for sqlfile_path in &sqlfile_paths {
            let sql_file = std::fs::File::open(sqlfile_path)?;
            sql_files.push(SqlFileWrapper(sql_file));
        }

        let mut partition_func : Arc<dyn PartitionFunc> = Arc::new(DefaultPartition{});

        if let Some(py_partition_func_spec_obj) = partition_func_spec_obj {
            partition_func = py_partition_func_spec_obj_to_rust(&py_partition_func_spec_obj, &columns)?;
        }

        let loader = ArrowLoader::new(sql_files, columns, thread_num, batch_data_threshold, compression_type, dialect, Some(partition_func));

        return Ok(Self { loader: Arc::new(Mutex::new(loader)) });
    }

    pub fn next_batch_data(&self, py : Python<'_>) -> anyhow::Result<Option<Vec<Vec<PyArray>>>> {
        let loader = self.loader.clone();
        py.allow_threads(move || -> anyhow::Result<Option<Vec<Vec<PyArray>>>> {
            loader.lock().unwrap().next_batch_data()
        })
    }
}