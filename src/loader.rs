use std::{collections::HashMap, io::Read, sync::{atomic::{AtomicBool, AtomicUsize}, Arc}, thread::{self, JoinHandle}, time};
use anyhow::{anyhow};
use arrow::{array::{Array, ArrayRef as ArrowArrayRef}, compute::{SortColumn, TakeOptions}};
use arrow_array::UInt32Array;
use crossbeam_channel::{Receiver, Sender};
use flate2::read::GzDecoder;
use pyo3_arrow::PyArray;
use sqlparser::{ast::{Insert, SetExpr, Statement, Values}, dialect::{self, Dialect}};

use crate::{arraybuilder, partition::{self, get_parition_key_from_first_val, DefaultPartition, PartitionFunc, PartitionKey}, pydebug, types::{self, ColumnArrStrDef}};

// struct SqlDataIter<T>
// where T : Into<Vec<u8>> + Send + 'static
// {
//     sql_datasets : Vec<T>,
//     idx : usize
// }

// impl <'a, T> SqlDataIter<'a, T> 
// where T : 'a + Into<Vec<u8>> + Send
// {
//     fn new(sql_datasets : &[T]) -> SqlDataIter<T> {
//         SqlDataIter {
//             sql_datasets : sql_datasets,
//             idx : 0,
//         }
//     }
// }

// impl <'a, T> Iterator for SqlDataIter<'a, T> 
// where T : 'a + Into<Vec<u8>> + Send
// {
//     type Item = &'a T;
    
//     fn next(&mut self) -> Option<Self::Item> {
//         if self.idx < self.sql_datasets.len() {
//             let res = &self.sql_datasets[self.idx];
//             self.idx += 1;
//             Some(res)
//         } else {
//             None
//         }
//     }
// }




//基本逻辑是只多算一个周期的数据，避免内存问题，周期计算方式再看
//当前算完了等待next处理的时候，可以继续算，如果没有取，任务就不进线程池
//线程池需要挂住以及等待，直到明确退出。
//线程池为每个generator独立线程池，根据用户自行设置决定。
//Loader本身不支持多线程的传递

/*
 * 基础逻辑为：
 *  1. 调用 load方法获取一个结果集，[]，一直调用直到为None
 *  2. 结束。
 * 
 * 内部逻辑：
 *  1. 第一次load时进行初始化。
 *  2. 初始化对线程池初始化。
 *  3. 创建处理任务给到线程池。
 *  4. 
 * 销毁逻辑：
 *  1. py对应对象持有这个generator
 *  2. 销毁时对线程暂停处理，对调用返回异常。
 *  3. 清理已经存在的内存数据
 */
struct ArrowLoader<T>
where T : Into<Vec<u8>> + Send + 'static
{
    sql_datasets_op : Option<Vec<T>>,
    batch_data_threshold : usize,
    thread_num : usize,
    columns: ColumnArrStrDef,
    partition_func_op : Option<Arc<dyn PartitionFunc>>,
    compression_type_op : Option<String>,
    dialect_op : Option<String>,
    state: AtomicUsize,
    res_rx_op : Option<Receiver<anyhow::Result<Vec<Vec<PyArray>>>>>,
    worker_thread_handlers : Vec<JoinHandle<()>>,
    collector_thread_handler : Vec<JoinHandle<()>>,
    sql_dataset_iter_thread_handler : Vec<JoinHandle<()>>,
}

const STATE_UNITIALIZED : usize = 0;
const STATE_INITIALIZING : usize = 10;
const STATE_LOADING: usize = 20;
const STATE_STOPPING : usize = 30;
const STATE_FINISHED : usize = 40;

impl <T> ArrowLoader <T>
where T : Into<Vec<u8>> + Send + Sync + 'static
{
    fn new(sql_datasets : Vec<T>, columns : ColumnArrStrDef,
        partition_func_op : Option<Arc<dyn PartitionFunc>>,
        compression_type_op : Option<String>,
        dialect_op : Option<String>,
        batch_data_threshold : usize,
        thread_num : usize,
    ) -> ArrowLoader<T> {
        let thread_num = if thread_num == 0 {
            //default thread num
            6
        } else {
            thread_num
        };

        ArrowLoader {
            sql_datasets_op : Some(sql_datasets),
            batch_data_threshold : batch_data_threshold,
            thread_num : thread_num,
            columns : columns,
            partition_func_op : partition_func_op,
            compression_type_op : compression_type_op,
            dialect_op : dialect_op,
            state : AtomicUsize::new(STATE_UNITIALIZED),
            res_rx_op : None,
            worker_thread_handlers : Vec::with_capacity(thread_num),
            collector_thread_handler : Vec::with_capacity(1),
            sql_dataset_iter_thread_handler : Vec::with_capacity(1),
        }
    }

    fn next_batch_data(&mut self) -> Option<anyhow::Result<Vec<Vec<PyArray>>>> {
        let cur_state = match self.state.compare_exchange(STATE_UNITIALIZED, STATE_INITIALIZING, std::sync::atomic::Ordering::Acquire, std::sync::atomic::Ordering::Relaxed) {
            Ok(_) => {
                self.init();
                STATE_LOADING
            },
            Err(v) => v,
        };

        match cur_state {
            STATE_INITIALIZED => {
                // start loading
                let _ = self.state.compare_exchange(STATE_INITIALIZED, STATE_LOADING, std::sync::atomic::Ordering::Acquire, std::sync::atomic::Ordering::Relaxed);
            },
            STATE_STOPPING => {
                return Some(Err(anyhow!("arrow loader is stopping")));
            },
            STATE_FINISHED => {
                return None;
            },
            _ => {
                return Some(Err(anyhow!("arrow loader ")));
            }
        }

        match self.res_rx_op.as_ref().unwrap().recv() {
            Ok(res) => {
                match res {
                    Ok(arr_refs_batch) => {
                        return Some(Ok(arr_refs_batch));
                    },
                    Err(e) => {
                        self.stop();
                        return Some(Err(e));
                    }
                }
            },
            Err(e) => {
                self.stop();
                return Some(Err(anyhow::Error::new(e)));
            }
        }
    }

    fn init(&mut self) {

        match self.state.compare_exchange(STATE_INITIALIZING, STATE_LOADING, std::sync::atomic::Ordering::Acquire, std::sync::atomic::Ordering::Relaxed) {
            Err(v) => {
                panic!("init when state is {:?}", v);
            },
            _ => (),
        };

        let (sql_dataset_tx, sql_dataset_rx) = crossbeam_channel::bounded::<anyhow::Result<T>>(self.thread_num);

        let (raw_res_tx, raw_res_rx) = crossbeam_channel::bounded::<anyhow::Result<HashMap<PartitionKey, Vec<ArrowArrayRef>>>>(self.thread_num);

        let (res_tx, res_rx) = crossbeam_channel::bounded::<anyhow::Result<Vec<Vec<PyArray>>>>(self.thread_num);
        self.res_rx_op = Some(res_rx.clone());

        //sql_dataset iterate and send to channel
        let thread_sql_dataset_tx = sql_dataset_tx.clone();
        let thread_sql_datasets = self.sql_datasets_op.replace(vec![]).unwrap();
        let sql_dataset_iter_thread_handler = thread::spawn(move || {
            for res_data in thread_sql_datasets {
                let _ = thread_sql_dataset_tx.send(Ok(res_data));
            }
        });
        self.sql_dataset_iter_thread_handler.push(sql_dataset_iter_thread_handler);

        
        for i in 0..self.thread_num {
            let thread_partition_func_op = self.partition_func_op.clone();
            let thread_compression_type_op = self.compression_type_op.clone();
            let thread_dialect_op = self.compression_type_op.clone();
            let thread_sql_dataset_rx = sql_dataset_rx.clone();
            let thread_raw_res_tx = raw_res_tx.clone();
            let thread_columns = self.columns.clone();
            let i_thread = i;
            let handler = thread::spawn(move || {
                Self::worker_thread_fn(
                    thread_sql_dataset_rx,
                    thread_raw_res_tx, thread_columns,
                    thread_partition_func_op,
                    thread_compression_type_op,
                    thread_dialect_op, i_thread
                );
            });
            self.worker_thread_handlers.push(handler);
        }

        
        let thread_raw_res_rx = raw_res_rx.clone();
        let thread_res_tx = res_tx.clone();
        let thread_batch_data_threshold = self.batch_data_threshold.clone();
        let thread_columns = self.columns.clone();
        let collector_handler = thread::spawn(move || {
            Self::collector_thread_fn(thread_batch_data_threshold, thread_columns, thread_raw_res_rx, thread_res_tx);
        });
    }

    fn stop(&mut self) {
        let state = self.state.load(std::sync::atomic::Ordering::Relaxed);
        let res = match state {
            STATE_INITIALIZED => {
                self.state.compare_exchange(STATE_INITIALIZED, STATE_STOPPING, std::sync::atomic::Ordering::Acquire, std::sync::atomic::Ordering::Relaxed)
            },
            STATE_LOADING => {
                self.state.compare_exchange(STATE_LOADING, STATE_STOPPING, std::sync::atomic::Ordering::Acquire, std::sync::atomic::Ordering::Relaxed)
            },
            _ => {
                Err(state)
            }
        };
        if res.is_err() {
            //do nothing
            return;
        }
        todo!()
    }

    fn collector_thread_fn(batch_data_threshold : usize, columns : ColumnArrStrDef, raw_res_rx : Receiver<Result<HashMap<PartitionKey, Vec<ArrowArrayRef>>, anyhow::Error>>, res_tx : Sender<Result<Vec<Vec<PyArray>>, anyhow::Error>>) {
        
        let mut hash_arr_refs_batch = HashMap::<PartitionKey, Vec<Vec<ArrowArrayRef>>>::new();
        let mut arrow_batch_data_len :usize = 0;
        for array_refs_res in raw_res_rx {
            match array_refs_res {
                Ok(hash_arr_refs) => {
                    for (partition_key, arr_refs) in hash_arr_refs {
                        if !hash_arr_refs_batch.contains_key(&partition_key) {
                            let arr_refs_batch = Vec::<Vec<ArrowArrayRef>>::with_capacity(12);
                            hash_arr_refs_batch.insert(partition_key.clone(), arr_refs_batch);
                        }
                        let arr_refs_batch = hash_arr_refs_batch.get_mut(&partition_key).unwrap();
                        arrow_batch_data_len += arr_refs.len();
                        arr_refs_batch.push(arr_refs);
                    }

                    if arrow_batch_data_len < batch_data_threshold || batch_data_threshold == 0 {
                        continue;
                    }

                    let res_pyarrays = generate_res_from_hash_arrs(&columns, &hash_arr_refs_batch);
                    
                    hash_arr_refs_batch = HashMap::<PartitionKey, Vec<Vec<ArrowArrayRef>>>::new();
                    arrow_batch_data_len = 0;
                    if let Err(_) = res_tx.send(res_pyarrays) {
                        break;
                    }
                },
                Err(e) => {
                    let _ = res_tx.send(Err(e));
                    break;
                }
            }
        }

        if arrow_batch_data_len > 0 {
            let res_pyarrays = generate_res_from_hash_arrs(&columns, &hash_arr_refs_batch);
            let _ = res_tx.send(res_pyarrays);
        }
        drop(res_tx);
    }


    fn worker_thread_fn(
        sql_dataset_rx : Receiver<Result<T, anyhow::Error>>,
        raw_res_tx : Sender<Result<HashMap<Vec<u8>, Vec<Arc<dyn Array>>>, anyhow::Error>>,
        columns : ColumnArrStrDef,
        partition_func_op : Option<Arc<dyn PartitionFunc>>,
        compression_type_op : Option<String>,
        dialect_op : Option<String>,
        i_thread : usize
    ) {
    
        let dialect_str = match dialect_op {
            Some(s) => s,
            None => "mysql".to_owned(),
        };
    
        let partition_func = match partition_func_op {
            Some(pf) => pf,
            None => Arc::new(DefaultPartition{}),
        };
    
        for res_sql_dataset in sql_dataset_rx {
            match res_sql_dataset {
                Ok(sql_dataset) => {
                    let sql_data_res = decompress_by_type(sql_dataset.into(), compression_type_op.clone(), i_thread);
                    if sql_data_res.is_err() {
                        let _ = raw_res_tx.send(Err(sql_data_res.err().unwrap()));
                        break;
                    } else {
                        let res_for_send : anyhow::Result<HashMap<PartitionKey, Vec<ArrowArrayRef>>> = (|| {
                            let sql_data = sql_data_res.unwrap();
                            let arr_refs = load_sql_data_to_arrref(&sql_data, &columns, &dialect_str, i_thread)?;
                            let partition_val_arr_refs = partition_func.transform(&arr_refs)?;
                            let indices = get_sorted_indices_from_multi_cols(&partition_val_arr_refs)?;
                            let ret = data_to_partitioned_arr_refs(&arr_refs, &partition_val_arr_refs, &indices);
                            ret
                        })();
                        let _ = raw_res_tx.send(res_for_send);
                    }
                },
                Err(e) => {
                    let _ = raw_res_tx.send(Err(e));
                },
            };
        }
        drop(raw_res_tx);
    }
    
}

fn generate_res_from_hash_arrs(columns : &ColumnArrStrDef, hash_arr_refs_batch : &HashMap::<PartitionKey, Vec<Vec<ArrowArrayRef>>>) -> anyhow::Result<Vec<Vec<PyArray>>> {
    let mut ret_pyarrs = Vec::<Vec<PyArray>>::with_capacity(hash_arr_refs_batch.len());
    for (_, arr_refs_batch) in hash_arr_refs_batch {
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

    Ok(ret_pyarrs)
}

fn parse_dialect(dialect : &str) -> anyhow::Result<Box<dyn Dialect>> {
    match dialect {
        "mysql" => Ok(Box::new(dialect::MySqlDialect{})),
        "postgresql" => Ok(Box::new(dialect::PostgreSqlDialect{})),
        _ => Err(anyhow!("not supported dialect"))
    }
}

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

fn decompress_by_type(sql_data : Vec<u8>, compression_type_op : Option<String>, i_thread : usize) -> anyhow::Result<Vec<u8>> {
    if let Some(compression_type) = compression_type_op {
        let decompress_start_time = time::Instant::now();
        let data_res = match compression_type.as_str() {
            "gzip" => {
                let mut decoder = GzDecoder::new(sql_data.as_slice());
                let mut buf = Vec::new();
                let _ = decoder.read_to_end(&mut buf)?;
                Ok(buf)
            },
            "snappy" => {
                let mut decoder = snap::read::FrameDecoder::new(sql_data.as_slice());
                let mut buf = Vec::new();
                let _ = decoder.read_to_end(&mut buf)?;
                Ok(buf)
            },
            _ => Err(anyhow!("not supported compression type"))
        };

        if data_res.is_ok() {
            pydebug!("thread(idx:{}) took {} seconds to decompress {} bytes size of {}-compressed data.", i_thread, decompress_start_time.elapsed().as_secs_f32(), sql_data.len(), compression_type.as_str());
        }

        data_res
    } else {
        Ok(sql_data)
    }
}


/**
 * columns
 * [
 *     index => (column_name,  data_type)
 * ]
 */
fn load_sql_data_to_arrref(sql_data : &Vec<u8>, columns : &ColumnArrStrDef, dialect_str : &str, idx_thread : usize) -> anyhow::Result<Vec<ArrowArrayRef>> {
    if sql_data.is_empty() || columns.is_empty() {
        return Err(anyhow!("sql_data is empty or columns is empty"));
    }

    let inner_parsing_building_start_time = time::Instant::now();

    let mut dt_vec = Vec::<&str>::with_capacity(columns.len());
    let mut column_name_to_outidx = HashMap::<String, usize>::with_capacity(columns.len());
    let mut i : usize = 0;
    for v in columns {
        dt_vec.push(&v.1);
        column_name_to_outidx.insert(v.0.clone(), i);
        i += 1;
    }


    let row_schema : types::RowSchema = dt_vec.try_into()?;

    let buffer = unsafe {
        std::str::from_utf8_unchecked(&sql_data)
    };
    let dia = parse_dialect(dialect_str)?;
    let mut sql_parser = sqlparser::parser::Parser::new(dia.as_ref());
    sql_parser = sql_parser.try_with_sql(&buffer)?;
  
    
    let mut val_idx_to_outidx = HashMap::<usize, usize>::with_capacity(columns.len());

    let mut expecting_statement_delimiter = false;

    let mut builders = row_schema.create_row_array_builders(10000);


    let mut total_seconds_for_parsing : f32 = 0.0;
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
        let parsing_start_time = time::Instant::now();
        let statement = sql_parser.parse_statement()?;
        total_seconds_for_parsing += parsing_start_time.elapsed().as_secs_f32();

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

    pydebug!("thread(idx:{}) took {} seconds to parsing data into insert statments.", idx_thread, total_seconds_for_parsing);
    pydebug!("thread(idx:{}) took {} seconds to parsing and building arrows.", idx_thread, inner_parsing_building_start_time.elapsed().as_secs_f32());
    Ok(arrays)
}

impl <T> Drop for ArrowLoader<T> 
where T : Into<Vec<u8>> + Send + 'static
{
    fn drop(&mut self) {
        todo!()
    }
}


impl <T> Iterator for ArrowLoader<T> 
where T : Into<Vec<u8>> + Send + 'static
{
    type Item = anyhow::Result<(usize, Vec<ArrowArrayRef>)>;

    fn next(&mut self) -> Option<Self::Item> {
        todo!()
    }
}