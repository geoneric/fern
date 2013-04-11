#include "ranally/operation/core/data_types.h"
#include <map>


namespace ranally {

DataTypes const DataTypes::UNKNOWN;
DataTypes const DataTypes::SCALAR(1 << DataType::DT_SCALAR);
DataTypes const DataTypes::POINT(1 << DataType::DT_POINT);
DataTypes const DataTypes::LINE(1 << DataType::DT_LINE);
DataTypes const DataTypes::POLYGON(1 << DataType::DT_POLYGON);
DataTypes const DataTypes::FEATURE(DataTypes::POINT | DataTypes::LINE |
    DataTypes::POLYGON);
DataTypes const DataTypes::ALL(DataTypes::SCALAR | DataTypes::FEATURE);


// These strings should match the ones used in the XML schema.
static std::map<String, DataTypes> data_type_by_string = {
    { "Scalar"        , DataTypes::SCALAR           },
    { "Point"         , DataTypes::POINT            },
    { "Line"          , DataTypes::LINE             },
    { "Polygon"       , DataTypes::POLYGON          },
    { "Feature"       , DataTypes::FEATURE          },
    { "All"           , DataTypes::ALL              }
};


static std::map<DataType, String> string_by_data_type = {
    { DataType::DT_SCALAR          , "Scalar"         },
    { DataType::DT_POINT           , "Point"          },
    { DataType::DT_LINE            , "Line"           },
    { DataType::DT_POLYGON         , "Polygon"        }
};


DataTypes DataTypes::from_string(
    String const& string)
{
    assert(!string.is_empty());
    assert(data_type_by_string.find(string) != data_type_by_string.end());
    return data_type_by_string[string];
}


std::vector<DataType> const DataTypes::DATA_TYPES = {
    DataType::DT_SCALAR,
    DataType::DT_POINT,
    DataType::DT_LINE,
    DataType::DT_POLYGON
};


static String to_string(
    DataType data_type)
{
    assert(string_by_data_type.find(data_type) != string_by_data_type.end());
    return string_by_data_type[data_type];
}


DataTypes::DataTypes()

    : FlagCollection<DataTypes, DataType, DataType::DT_LAST_DATA_TYPE + 1>()

{
}


String DataTypes::to_string() const
{
    assert(DataTypes::DATA_TYPES.size() == DT_LAST_DATA_TYPE + 1);
    std::vector<String> strings;

    for(DataType data_type: DataTypes::DATA_TYPES) {
        if(test(data_type)) {
            strings.push_back(ranally::to_string(data_type));
        }
    }

    if(strings.empty()) {
        strings.push_back("?");
    }

    return join(strings, "|");
}


DataTypes operator|(
    DataTypes const& lhs,
    DataTypes const& rhs)
{
    DataTypes result(lhs);
    result |= rhs;
    return result;
}


std::ostream& operator<<(
    std::ostream& stream,
    DataTypes const& flags)
{
    stream << flags.to_string();
    return stream;
}

} // namespace ranally
