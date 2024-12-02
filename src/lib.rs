mod types;
mod arraybuilder;
mod partition;


use std::collections::HashMap;
use std::io::Read;
use std::sync::Arc;
use std::thread;
use std::time::Instant;

use anyhow::{anyhow};
use arrow::array::{Array, ArrayRef as ArrowArrayRef};
use arrow::compute::{SortColumn, TakeOptions};
use arrow_array::UInt32Array;
use flate2::read::GzDecoder;
use mimalloc::MiMalloc;
use partition::{get_parition_key_from_first_val, py_partition_func_spec_obj_to_rust, DefaultPartition, PartitionFunc, PartitionKey};
use pyo3::{iter, prelude::*};
use pyo3_arrow::error::PyArrowResult;
use pyo3_arrow::{PyArray, PyChunkedArray};
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
    Ok(())
}

#[pyfunction]
#[pyo3(signature = (sql_paths, columns, partition_func_spec_obj=None, compression_type=None, dialect=None))]
fn load_sqls(sql_paths : Vec<String>, columns: ColumnArrStrDef, partition_func_spec_obj : Option<PyObject>, compression_type : Option<String>, dialect : Option<String>) -> anyhow::Result<Vec<Vec<PyArray>>> {
    if sql_paths.is_empty() {
        return Err(anyhow!("sql_paths is empty"));
    }

    let mut sql_dataset = Vec::with_capacity(sql_paths.len());
    for sql_path in sql_paths {
        let buffer = std::fs::read(sql_path)?;
        sql_dataset.push(buffer);
    }

    load_sqls_with_dataset(sql_dataset, columns, partition_func_spec_obj, compression_type, dialect)
}

#[pyfunction]
#[pyo3(signature = (sql_dataset, columns, partition_func_spec_obj=None, compression_type=None, dialect=None))]
fn load_sqls_with_dataset(sql_dataset : Vec<Vec<u8>>, columns: ColumnArrStrDef, partition_func_spec_obj : Option<PyObject>, compression_type : Option<String>, dialect : Option<String>) -> anyhow::Result<Vec<Vec<PyArray>>> {
    if sql_dataset.is_empty() {
        return Err(anyhow!("sql_dataset is empty"));
    }

    let mut partition_func : Arc<dyn PartitionFunc> = Arc::new(DefaultPartition{});
    let mut is_have_partition_func = false;

    if let Some(py_partition_func_spec_obj) = partition_func_spec_obj {
        partition_func = py_partition_func_spec_obj_to_rust(&py_partition_func_spec_obj, &columns)?;
        is_have_partition_func = true;
    }

    let data = if !is_have_partition_func {
        match load_without_partition(sql_dataset, columns, compression_type, dialect) {
            Ok(pyarrs) => PyArrowResult::Ok(pyarrs),
            Err(e) => Err(pyo3_arrow::error::PyArrowError::PyErr(
                    pyo3::exceptions::PyRuntimeError::new_err(e.to_string())
                )),
        }
    } else {
        match load_with_partition(sql_dataset, columns, partition_func, compression_type, dialect) {
            Ok(pyarrs) => PyArrowResult::Ok(pyarrs),
            Err(e) => Err(pyo3_arrow::error::PyArrowError::PyErr(
                pyo3::exceptions::PyRuntimeError::new_err(e.to_string())
            )),
        }
    }?;

    Ok(data)
}

fn decompress_by_type(sql_data : Vec<u8>, compression_type_op : Option<String>) -> anyhow::Result<Vec<u8>> {
    if let Some(compression_type) = compression_type_op {
        match compression_type.as_str() {
            "gzip" => {
                    let mut decoder = GzDecoder::new(sql_data.as_slice());
                    let mut buf = Vec::new();
                    let _ = decoder.read_to_end(&mut buf);
                    Ok(buf)
            },
            _ => Err(anyhow!("not supported compression type"))
        }
    } else {
        Ok(sql_data)
    }
}

fn parse_dialect(dialect_op : &Option<String>) -> anyhow::Result<Box<dyn Dialect>> {
    if let Some(dialect_str) = dialect_op {
        match dialect_str.as_str() {
            "mysql" => Ok(Box::new(dialect::MySqlDialect{})),
            _ => Err(anyhow!("not supported dialect"))
        }
    } else {
        Ok(Box::new(dialect::MySqlDialect{}))
    }
}

fn load_without_partition(sql_dataset : Vec<Vec<u8>>, columns : ColumnArrStrDef, compression_type_op : Option<String>, dialect_op : Option<String>) -> anyhow::Result<Vec<Vec<PyArray>>> {
    let sql_dataset_len = sql_dataset.len();
    let _dia = parse_dialect(&dialect_op)?;

    let (tx, rx) = crossbeam_channel::bounded::<anyhow::Result<(usize, Vec<ArrowArrayRef>)>>(sql_dataset.len());

    let mut handlers = Vec::with_capacity(sql_dataset.len());
    let mut i : usize = 0;
    for sql_data in sql_dataset {
        let tx_thread = tx.clone();
        let columns_thread = columns.clone();
        let dialect_op_thread = dialect_op.clone();
        let compression_type_op_thread = compression_type_op.clone();
        let i_thread = i;
        i += 1;

        let handler = thread::spawn(move || {
            let sql_data_res = decompress_by_type(sql_data, compression_type_op_thread);
            if sql_data_res.is_err() {
                let _ = tx_thread.send(Err(sql_data_res.err().unwrap()));
            } else {
                let sql_data = sql_data_res.unwrap();
                match load_sql_data_to_arrref(&sql_data, columns_thread, dialect_op_thread,i_thread) {
                    Ok(arr_refs) => {
                        let _ = tx_thread.send(Ok((i_thread, arr_refs)));
                    },
                    Err(e) => {
                        let _ = tx_thread.send(Err(e));
                    }
                }
            }
            drop(tx_thread)
        });



        handlers.push(handler);
    }
    drop(tx);

    let mut ret_pyarrs = Vec::<Vec<PyArray>>::with_capacity(sql_dataset_len);
    for _ in 0..sql_dataset_len {
        ret_pyarrs.push(Vec::<PyArray>::with_capacity(columns.len()));
    }

    let mut res = anyhow::Ok(());
    for array_refs_res in rx {
        match array_refs_res {
            Ok((i, arr_refs)) => {
                let pyarrs = ret_pyarrs.get_mut(i).unwrap();
                for arr_ref in arr_refs {
                    pyarrs.push(PyArray::from_array_ref(arr_ref));
                }
            },
            Err(e) => {
                res = Err(e);
                break;
            }
        }
    }

    for handler in handlers {
        let _ = handler.join();
    }
    
    if res.is_err() {
        return Err(res.err().unwrap());
    }
    

    return Ok(ret_pyarrs);
}


fn load_with_partition(sql_dataset : Vec<Vec<u8>>, columns : ColumnArrStrDef, partition_func : Arc<dyn PartitionFunc>, compression_type_op : Option<String>, dialect_op : Option<String>) -> anyhow::Result<Vec<Vec<PyArray>>> {
    fn get_sorted_indices_from_multi_cols(arr_refs : &Vec<ArrowArrayRef>) -> anyhow::Result<UInt32Array> {
        let mut sort_cols = Vec::<SortColumn>::with_capacity(arr_refs.len());
        for arr_ref in arr_refs {
            let sort_col = SortColumn {
                values : arr_ref.clone(),
                options: None,
            };
            sort_cols.push(sort_col);
        }

        return Ok(arrow::compute::lexsort_to_indices(&sort_cols, None)?);
    }

    fn data_to_partitioned_arr_refs(arr_refs: &Vec<ArrowArrayRef>, partition_val_arr_refs: &Vec<ArrowArrayRef>, sorted_indices : &UInt32Array) -> anyhow::Result<HashMap<PartitionKey, Vec<ArrowArrayRef>>> {
        let take_opt = TakeOptions{check_bounds:true};
        let sorted_arr_refs = arrow::compute::take_arrays(&arr_refs, &sorted_indices, Some(take_opt.clone()))?;
        let sorted_partition_val_arr_refs = arrow::compute::take_arrays(&partition_val_arr_refs, &sorted_indices, Some(take_opt.clone()))?;
        let partitions = arrow::compute::partition(&sorted_partition_val_arr_refs)?;

        let mut res = HashMap::<PartitionKey, Vec<ArrowArrayRef>>::with_capacity(partitions.len());

        for (_, r) in partitions.ranges().iter().enumerate() {
            

            let mut partitioned_arr_refs = Vec::<ArrowArrayRef>::with_capacity(sorted_arr_refs.len());
            for arr_ref in &sorted_arr_refs {
                let partitioned_arr_ref = arr_ref.slice(r.start, r.end - r.start);
                partitioned_arr_refs.push(partitioned_arr_ref);
            }

            let mut partitioned_val_arr_refs = Vec::<ArrowArrayRef>::with_capacity(sorted_partition_val_arr_refs.len());
            for arr_ref in &sorted_partition_val_arr_refs {
                let partitioned_val_arr_ref = arr_ref.slice(r.start, r.end - r.start);
                partitioned_val_arr_refs.push(partitioned_val_arr_ref);
            }

            let partition_key = get_parition_key_from_first_val(&partitioned_val_arr_refs)?;

            res.insert(partition_key, partitioned_arr_refs);
        }

        

        return Ok(res);
    }

    
    let sql_dataset_len = sql_dataset.len();
    let _dia = parse_dialect(&dialect_op)?;
    let (tx, rx) = crossbeam_channel::bounded::<anyhow::Result<HashMap<PartitionKey, Vec<ArrowArrayRef>>>>(sql_dataset_len);

    let mut handlers = Vec::with_capacity(sql_dataset_len);
    let mut i : usize = 0;
    for sql_data in sql_dataset {
        let tx_thread = tx.clone();
        let columns_thread = columns.clone();
        let partition_func_thread = partition_func.clone();
        let dialect_op_thread = dialect_op.clone();
        let compression_type_op_thread = compression_type_op.clone();
        let i_thread = i;
        i += 1;

        let handler = thread::spawn(move || {

            let res_for_send : anyhow::Result<HashMap<PartitionKey, Vec<ArrowArrayRef>>> = (move || {
                let sql_data = decompress_by_type(sql_data, compression_type_op_thread)?;
                let arr_refs = load_sql_data_to_arrref(&sql_data, columns_thread, dialect_op_thread, i_thread)?;

                let mut is_debug = false;
                match std::env::var("SQL2ARROW_DEBUG") {
                    Ok(value) => is_debug = true,
                    _ => {}
                }
                let partition_start_time = Instant::now();
                let partition_val_arr_refs = partition_func_thread.transform(&arr_refs)?;
                let indices = get_sorted_indices_from_multi_cols(&partition_val_arr_refs)?;
                let ret = data_to_partitioned_arr_refs(&arr_refs, &partition_val_arr_refs, &indices);
                if is_debug {
                    print!("thread idx: {} partition. {:?}\n", i_thread, partition_start_time.elapsed());
                }
                ret
            })();

            match res_for_send {
                Ok(hash_arr_refs) => {
                    let _ = tx_thread.send(Ok(hash_arr_refs));
                },
                Err(e) => {
                    let _ = tx_thread.send(Err(e));
                }
            }
            drop(tx_thread)
        });


        handlers.push(handler);
    }
    drop(tx);
    
    let mut hash_arr_refs_batch = HashMap::<PartitionKey, Vec<Vec<ArrowArrayRef>>>::new();

    let mut res = anyhow::Ok(());
    for array_refs_res in rx {
        match array_refs_res {
            Ok(hash_arr_refs) => {
                for (partition_key, arr_refs) in hash_arr_refs {
                    if !hash_arr_refs_batch.contains_key(&partition_key) {
                        let arr_refs_batch = Vec::<Vec<ArrowArrayRef>>::with_capacity(sql_dataset_len);
                        hash_arr_refs_batch.insert(partition_key.clone(), arr_refs_batch);
                    }
                    let arr_refs_batch = hash_arr_refs_batch.get_mut(&partition_key).unwrap();
                    arr_refs_batch.push(arr_refs);
                }
            },
            Err(e) => {
                res = Err(e);
                break;
            }
        }
    }

    for handler in handlers {
        let _ = handler.join();
    }

    if res.is_err() {
        return Err(res.err().unwrap());
    }

    let mut is_debug = false;
    match std::env::var("SQL2ARROW_DEBUG") {
        Ok(value) => is_debug = true,
        _ => {}
    }

    let combine_partition_start_time = Instant::now();

    let mut ret_pyarrs = Vec::<Vec<PyArray>>::with_capacity(hash_arr_refs_batch.len());
    for (_, arr_refs_batch) in &hash_arr_refs_batch {
        let mut vertical_arr_refs = vec![Vec::<ArrowArrayRef>::with_capacity(hash_arr_refs_batch.len()); columns.len()];
        for arr_refs in arr_refs_batch {
            for (idx, arr_ref) in arr_refs.iter().enumerate() {
                vertical_arr_refs.get_mut(idx).unwrap().push(arr_ref.clone());
            }
        }

        let mut new_arr_refs = Vec::<PyArray>::with_capacity(columns.len());
        for col_arr_refs in vertical_arr_refs {
            let arr_refs_for_concat : Vec<&dyn Array> = col_arr_refs.iter().map(|arc| arc.as_ref()).collect();
            let arr_ref = arrow::compute::concat(arr_refs_for_concat.as_slice())?;
            new_arr_refs.push(PyArray::from_array_ref(arr_ref));
        }

        ret_pyarrs.push(new_arr_refs);
    }
    
    if is_debug {
        print!("combine partition. {:?}\n", combine_partition_start_time.elapsed());
    }

    return Ok(ret_pyarrs);
}
/**
 * columns
 * [
 *     index => (column_name,  data_type)
 * ]
 */
fn load_sql_data_to_arrref(sql_data : &Vec<u8>, columns : ColumnArrStrDef, dialect_op : Option<String>, idx_thread : usize) -> anyhow::Result<Vec<ArrowArrayRef>> {
    use std::env;
    let mut is_debug = false;
    match env::var("SQL2ARROW_DEBUG") {
        Ok(value) => is_debug = true,
        _ => {}
    }

    if sql_data.is_empty() || columns.is_empty() {
        return Err(anyhow!("sql_data is empty or columns is empty"));
    }

    let mut dt_vec = Vec::<&str>::with_capacity(columns.len());
    let mut column_name_to_outidx = HashMap::<String, usize>::with_capacity(columns.len());
    let mut i : usize = 0;
    for v in &columns {
        dt_vec.push(&v.1);
        column_name_to_outidx.insert(v.0.clone(), i);
        i += 1;
    }


    let row_schema : types::RowSchema = dt_vec.try_into()?;

    let parse_sql_start_time = Instant::now();
    let buffer = unsafe {
        std::str::from_utf8_unchecked(&sql_data)
    };
    let dia = parse_dialect(&dialect_op)?;
    let mut sql_parser = sqlparser::parser::Parser::new(dia.as_ref());
    sql_parser = sql_parser.try_with_sql(&buffer)?;
  
    if is_debug {
        print!("thread idx: {} parse sql. {:?}\n", idx_thread, parse_sql_start_time.elapsed());
    }
    
    let mut val_idx_to_outidx = HashMap::<usize, usize>::with_capacity(columns.len());

    let mut expecting_statement_delimiter = false;

    let mut builders = row_schema.create_row_array_builders(10000);
    //loop statement
    loop {
        while sql_parser.consume_token(&sqlparser::tokenizer::Token::SemiColon) {
            expecting_statement_delimiter = false;
        }

        match sql_parser.peek_token().token {
            sqlparser::tokenizer::Token::EOF => break,

            // end of statement
            sqlparser::tokenizer::Token::Word(word) => {
                if expecting_statement_delimiter && word.keyword == sqlparser::keywords::Keyword::END {
                    break;
                }
            }
            _ => {}
        }

        if expecting_statement_delimiter {
            return sql_parser.expected("end of statement", sql_parser.peek_token())?;
        }

        let statement = sql_parser.parse_statement()?;
        if val_idx_to_outidx.is_empty() {
            match &statement {
                Statement::Insert(Insert{columns, ..}) => {
                    if !columns.is_empty() {
                        //match the column names
                        let mut val_idx = 0;
                        for col in columns {
                            if column_name_to_outidx.contains_key(col.value.as_str()) {
                                val_idx_to_outidx.insert(val_idx, column_name_to_outidx.get(col.value.as_str()).unwrap().clone());
                                column_name_to_outidx.remove(col.value.as_str());
                            }
                            val_idx += 1;
                        }
    
                        if !column_name_to_outidx.is_empty() {
                            let not_exists_columns_name : Vec<String> = column_name_to_outidx.keys().cloned().collect();
                            return Err(anyhow!(format!("these columns: {} not exists", not_exists_columns_name.join(","))));
                        }
                    } else {
                        //Insert Into xxx VALUES(xxx,xxx)
                        //no columns
                        for (_, outidx) in column_name_to_outidx.iter() {
                            val_idx_to_outidx.insert(outidx.clone(), outidx.clone());
                        }
                    }
                },
                _ => (),
            }
        }

        
        match statement {
            Statement::Insert(Insert{source, ..}) => {
                match source.as_ref().unwrap().body.as_ref() {
                    SetExpr::Values(Values{  rows, .. }) => {
                        for row in rows {
                            for (val_idx, outidx) in val_idx_to_outidx.iter() {
                                let b = builders.get_mut(outidx.clone()).unwrap();
                                let dt = row_schema.get(outidx.clone()).unwrap();
                                let expr = row.get(val_idx.clone()).unwrap();
                                
                                arraybuilder::append_value_to_builder(b, dt, expr)?;
                            }
                        }
                    },
                    _ => (),
                };
            },
            _ => (),
        }
    } //end of loop

    let mut arrays = Vec::<ArrowArrayRef>::with_capacity(builders.len());
    for mut b in builders {
        let arr_ref = b.finish();
        arrays.push(arr_ref);
    }

    Ok(arrays)
}