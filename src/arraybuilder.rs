use arrow::compute::kernels::cast_utils::parse_decimal;
use arrow::{array as arrow_array, datatypes::Decimal128Type};
use arrow::datatypes::{i256, DataType as ArrowDataType, Decimal256Type};
use sqlparser::ast::{Expr, UnaryOperator, Value};

#[inline]
pub fn append_value_to_builder(builder : &mut Box<dyn arrow_array::ArrayBuilder>, datatype : &ArrowDataType, expr : &Expr) -> anyhow::Result<()> {

    match datatype {
        ArrowDataType::Int8 => {
            let b = builder.as_any_mut().downcast_mut::<arrow_array::Int8Builder>().unwrap();
            match expr {
                Expr::UnaryOp {
                    op : UnaryOperator::Minus,
                    expr 
                } => {
                    match expr.as_ref() {
                        Expr::Value(Value::Number(num, _)) => {
                            let v : i8 = -num.parse()?;
                            b.append_value(v);
                        },
                        _ => return Err(arrow_schema::ArrowError::CastError(format!("not allowed Expr: {:?} to {:?}", expr, datatype)).into())
                    }
                },
                Expr::Value(Value::Number(num, _)) => {
                    let v : i8 = num.parse()?;
                    b.append_value(v);
                },
                Expr::Value(Value::Null) => b.append_null(),
                _ => return Err(arrow_schema::ArrowError::CastError(format!("not allowed Expr: {:?} to {:?}", expr, datatype)).into())
            }
        },
        //integer
        ArrowDataType::Int16 => {
            let b = builder.as_any_mut().downcast_mut::<arrow_array::Int16Builder>().unwrap();
            match expr {
                Expr::UnaryOp {
                    op : UnaryOperator::Minus,
                    expr 
                } => {
                    match expr.as_ref() {
                        Expr::Value(Value::Number(num, _)) => {
                            let v : i16 = -num.parse()?;
                            b.append_value(v);
                        },
                        _ => return Err(arrow_schema::ArrowError::CastError(format!("not allowed Expr: {:?} to {:?}", expr, datatype)).into())
                    }
                },
                Expr::Value(Value::Number(num, _)) => {
                    let v : i16 = num.parse()?;
                    b.append_value(v);
                },
                Expr::Value(Value::Null) => b.append_null(),
                _ => return Err(arrow_schema::ArrowError::CastError(format!("not allowed Expr: {:?} to {:?}", expr, datatype)).into())
            }
        },
        ArrowDataType::Int32 => {
            let b = builder.as_any_mut().downcast_mut::<arrow_array::Int32Builder>().unwrap();
            match expr {
                Expr::UnaryOp {
                    op : UnaryOperator::Minus,
                    expr 
                } => {
                    match expr.as_ref() {
                        Expr::Value(Value::Number(num, _)) => {
                            let v : i32 = -num.parse()?;
                            b.append_value(v);
                        },
                        _ => return Err(arrow_schema::ArrowError::CastError(format!("not allowed Expr: {:?} to {:?}", expr, datatype)).into())
                    }
                },
                Expr::Value(Value::Number(num, _)) => {
                    let v : i32 = num.parse()?;
                    b.append_value(v);
                },
                Expr::Value(Value::Null) => b.append_null(),
                _ => return Err(arrow_schema::ArrowError::CastError(format!("not allowed Expr: {:?} to {:?}", expr, datatype)).into())
            }
        },
        ArrowDataType::Int64 => {
            let b = builder.as_any_mut().downcast_mut::<arrow_array::Int64Builder>().unwrap();
            match expr {
                Expr::UnaryOp {
                    op : UnaryOperator::Minus,
                    expr 
                } => {
                    match expr.as_ref() {
                        Expr::Value(Value::Number(num, _)) => {
                            let v : i64 = -num.parse()?;
                            b.append_value(v);
                        },
                        _ => return Err(arrow_schema::ArrowError::CastError(format!("not allowed Expr: {:?} to {:?}", expr, datatype)).into())
                    }
                },
                Expr::Value(Value::Number(num, _)) => {
                    let v : i64 = num.parse()?;
                    b.append_value(v);
                },
                Expr::Value(Value::Null) => b.append_null(),
                _ => return Err(arrow_schema::ArrowError::CastError(format!("not allowed Expr: {:?} to {:?}", expr, datatype)).into())
            }
        },
        ArrowDataType::UInt8 => {
            let b = builder.as_any_mut().downcast_mut::<arrow_array::UInt8Builder>().unwrap();
            match expr {
                Expr::Value(Value::Number(num, _)) => {
                    let v : u8 = num.parse()?;
                    b.append_value(v);
                },
                Expr::Value(Value::Null) => b.append_null(),
                _ => return Err(arrow_schema::ArrowError::CastError(format!("not allowed Expr: {:?} to {:?}", expr, datatype)).into())
            }
        },
        ArrowDataType::UInt16 => {
            let b = builder.as_any_mut().downcast_mut::<arrow_array::UInt16Builder>().unwrap();
            match expr {
                Expr::Value(Value::Number(num, _)) => {
                    let v : u16 = num.parse()?;
                    b.append_value(v);
                },
                Expr::Value(Value::Null) => b.append_null(),
                _ => return Err(arrow_schema::ArrowError::CastError(format!("not allowed Expr: {:?} to {:?}", expr, datatype)).into())
            }
        },
        ArrowDataType::UInt32 => {
            let b = builder.as_any_mut().downcast_mut::<arrow_array::UInt32Builder>().unwrap();
            match expr {
                Expr::Value(Value::Number(num, _)) => {
                    let v : u32 = num.parse()?;
                    b.append_value(v);
                },
                Expr::Value(Value::Null) => b.append_null(),
                _ => return Err(arrow_schema::ArrowError::CastError(format!("not allowed Expr: {:?} to {:?}", expr, datatype)).into())
            }
        },
        ArrowDataType::UInt64 => {
            let b = builder.as_any_mut().downcast_mut::<arrow_array::UInt64Builder>().unwrap();
            match expr {
                Expr::Value(Value::Number(num, _)) => {
                    let v : u64 = num.parse()?;
                    b.append_value(v);
                },
                Expr::Value(Value::Null) => b.append_null(),
                _ => return Err(arrow_schema::ArrowError::CastError(format!("not allowed Expr: {:?} to {:?}", expr, datatype)).into())
            }
        },

        //float
        ArrowDataType::Float32 => {
            let b = builder.as_any_mut().downcast_mut::<arrow_array::Float32Builder>().unwrap();
            match expr {
                Expr::UnaryOp {
                    op : UnaryOperator::Minus,
                    expr 
                } => {
                    match expr.as_ref() {
                        Expr::Value(Value::Number(num, _)) => {
                            let v : f32 = -num.parse()?;
                            b.append_value(v);
                        },
                        _ => return Err(arrow_schema::ArrowError::CastError(format!("not allowed Expr: {:?} to {:?}", expr, datatype)).into())
                    }
                },
                Expr::Value(Value::Number(num, _)) => {
                    let v : f32 = num.parse()?;
                    b.append_value(v);
                },
                Expr::Value(Value::Null) => b.append_null(),
                _ => return Err(arrow_schema::ArrowError::CastError(format!("not allowed Expr: {:?} to {:?}", expr, datatype)).into())
            }
        },
        ArrowDataType::Float64 => {
            let b = builder.as_any_mut().downcast_mut::<arrow_array::Float64Builder>().unwrap();
            match expr {
                Expr::UnaryOp {
                    op : UnaryOperator::Minus,
                    expr 
                } => {
                    match expr.as_ref() {
                        Expr::Value(Value::Number(num, _)) => {
                            let v : f64 = -num.parse()?;
                            b.append_value(v);
                        },
                        _ => return Err(arrow_schema::ArrowError::CastError(format!("not allowed Expr: {:?} to {:?}", expr, datatype)).into())
                    }
                },
                Expr::Value(Value::Number(num, _)) => {
                    let v : f64 = num.parse()?;
                    b.append_value(v);
                },
                Expr::Value(Value::Null) => b.append_null(),
                _ => return Err(arrow_schema::ArrowError::CastError(format!("not allowed Expr: {:?} to {:?}", expr, datatype)).into())
            }
        },

        //decimal128
        ArrowDataType::Decimal128(precision, scale) => {
            let b = builder.as_any_mut().downcast_mut::<arrow_array::Decimal128Builder>().unwrap();
            match expr {
                Expr::UnaryOp {
                    op : UnaryOperator::Minus,
                    expr 
                } => {
                    match expr.as_ref() {
                        Expr::Value(Value::Number(num, _)) => {
                            let mut num_str = String::with_capacity(num.len()+1);
                            num_str.push('-');
                            num_str.push_str(num);
                            let v : i128 = parse_decimal::<Decimal128Type>(&num_str, precision.clone(), scale.clone())?;
                            b.append_value(v);
                        },
                        _ => return Err(arrow_schema::ArrowError::CastError(format!("not allowed Expr: {:?} to {:?}", expr, datatype)).into())
                    }
                },
                Expr::Value(Value::Number(num, _)) => {
                    
                    let v : i128 = parse_decimal::<Decimal128Type>(num, precision.clone(), scale.clone())?;
                    b.append_value(v);
                },
                Expr::Value(Value::Null) => b.append_null(),
                _ => return Err(arrow_schema::ArrowError::CastError(format!("not allowed Expr: {:?} to {:?}", expr, datatype)).into())
            }
        }

        //decimal256
        ArrowDataType::Decimal256(precision, scale) => {
            let b = builder.as_any_mut().downcast_mut::<arrow_array::Decimal256Builder>().unwrap();
            match expr {
                Expr::UnaryOp {
                    op : UnaryOperator::Minus,
                    expr 
                } => {
                    match expr.as_ref() {
                        Expr::Value(Value::Number(num, _)) => {
                            let mut num_str = String::with_capacity(num.len()+1);
                            num_str.push('-');
                            num_str.push_str(num);
                            let v : i256 = parse_decimal::<Decimal256Type>(&num_str, precision.clone(), scale.clone())?;
                            b.append_value(v);
                        },
                        _ => return Err(arrow_schema::ArrowError::CastError(format!("not allowed Expr: {:?} to {:?}", expr, datatype)).into())
                    }
                },
                Expr::Value(Value::Number(num, _)) => {
                    
                    let v : i256 = parse_decimal::<Decimal256Type>(num, precision.clone(), scale.clone())?;
                    b.append_value(v);
                },
                Expr::Value(Value::Null) => b.append_null(),
                _ => return Err(arrow_schema::ArrowError::CastError(format!("not allowed Expr: {:?} to {:?}", expr, datatype)).into())
            }
        }

        ArrowDataType::Boolean => {
            let b = builder.as_any_mut().downcast_mut::<arrow_array::BooleanBuilder>().unwrap();
            match expr {
                Expr::Value(Value::Boolean(bool)) => {
                    b.append_value(bool.clone());
                },
                Expr::Value(Value::SingleQuotedString(v)) => {
                    if v.eq_ignore_ascii_case("true") {
                        b.append_value(true);
                    } else if v.eq_ignore_ascii_case("false") {
                        b.append_value(false);
                    } else {
                        return Err(arrow_schema::ArrowError::CastError(format!("not allowed str value: {:?} to {:?}", v, datatype)).into())
                    }
                },
                Expr::Value(Value::DoubleQuotedString(v)) => {
                    if v.eq_ignore_ascii_case("true") {
                        b.append_value(true);
                    } else if v.eq_ignore_ascii_case("false") {
                        b.append_value(false);
                    } else {
                        return Err(arrow_schema::ArrowError::CastError(format!("not allowed str value: {:?} to {:?}", v, datatype)).into())
                    }
                },
                Expr::Value(Value::Null) => b.append_null(),
                _ => return Err(arrow_schema::ArrowError::CastError(format!("not allowed Expr: {:?} to {:?}", expr, datatype)).into())
            }
        },

        ArrowDataType::Utf8 => {
            let b = builder.as_any_mut().downcast_mut::<arrow_array::StringBuilder>().unwrap();
            match expr {
                Expr::Value(Value::SingleQuotedString(v)) => {
                    b.append_value(v);
                },
                Expr::Value(Value::DoubleQuotedString(v)) => {
                    b.append_value(v);
                },
                Expr::Value(Value::Null) => b.append_null(),
                _ => return Err(arrow_schema::ArrowError::CastError(format!("not allowed Expr: {:?} to {:?}", expr, datatype)).into())
            }
        }
        _ => {
            return Err(arrow_schema::ArrowError::NotYetImplemented(format!("data type {:?} not implemented yet", datatype)).into());
        },
    }

    Ok(())
}