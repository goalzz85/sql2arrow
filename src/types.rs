use std::ops::Deref;

use arrow::datatypes::DataType as ArrowDataType;

//column_name, data_type
pub type ColumnStrDef = (String, String);
pub type ColumnArrStrDef = Vec<ColumnStrDef>;

pub struct RowSchema(Vec<ArrowDataType>);

impl RowSchema {
    fn new(datatypes : Vec<ArrowDataType>) -> RowSchema {
        assert!(!datatypes.is_empty(), "arrow datatypes for RowSchema is empty");
        RowSchema(datatypes)
    }

    pub fn create_row_array_builders(&self, capacity : usize) -> Vec<Box<dyn arrow::array::ArrayBuilder + 'static>> {
        let mut res = Vec::<Box<dyn arrow::array::ArrayBuilder>>::with_capacity(self.len());
        for dt in &self.0 {
            res.push(arrow::array::make_builder(dt, capacity));
        }
        res
    }
}


impl Deref for RowSchema {
    type Target = Vec<ArrowDataType>;
    
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}


impl TryFrom<Vec<&str>> for RowSchema {    
    type Error = arrow_schema::ArrowError;
    
    fn try_from(value: Vec<&str>) -> Result<Self, Self::Error> {


        let mut vec_datatypes = Vec::<ArrowDataType>::with_capacity(value.len());
        for dt_str in value {
            let dt : ArrowDataType = dt_str.parse()?;
            if !is_allowed_datatype(&dt) {
                return Err(arrow_schema::ArrowError::NotYetImplemented(format!("not allowed datatype {:?}", &dt)));
            }
            vec_datatypes.push(dt);
        }

        Ok(RowSchema::new(vec_datatypes))
    }
}

fn is_allowed_datatype(dt : &ArrowDataType) -> bool {
    match dt {
        ArrowDataType::Int8 | ArrowDataType::Int16 | ArrowDataType::Int32 | ArrowDataType::Int64
        | ArrowDataType::UInt8 | ArrowDataType::UInt16 | ArrowDataType::UInt32 | ArrowDataType::UInt64 => true,
        
        ArrowDataType::Float32 | ArrowDataType::Float64 => true,

        ArrowDataType::Decimal128(_,_) => true,

        ArrowDataType::Boolean => true,

        ArrowDataType::Utf8 => true,

        _ => false,
    }
}