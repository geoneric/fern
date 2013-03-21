#include "ranally/operation/data_types.h"
#include <map>


namespace ranally {

// TODO Refactor.
DataTypes const DataTypes::UNKNOWN;
DataTypes const DataTypes::SCALAR({ detail::DataType::DT_SCALAR });
DataTypes const DataTypes::POINT({ detail::DataType::DT_POINT });
DataTypes const DataTypes::LINE({ detail::DataType::DT_LINE });
DataTypes const DataTypes::POLYGON({ detail::DataType::DT_POLYGON });
DataTypes const DataTypes::FEATURE({ detail::DataType::DT_POINT, detail::DataType::DT_LINE, detail::DataType::DT_POLYGON });
DataTypes const DataTypes::ALL({ detail::DataType::DT_SCALAR, detail::DataType::DT_POINT, detail::DataType::DT_LINE, detail::DataType::DT_POLYGON } );
DataTypes const DataTypes::DEPENDS_ON_INPUT({ detail::DataType::DT_DEPENDS_ON_INPUT });


// These strings should match the ones used in the XML schema.
static std::map<String, DataTypes> data_type_by_string = {
    { "Scalar"        , DataTypes::SCALAR           },
    { "Point"         , DataTypes::POINT            },
    { "Line"          , DataTypes::LINE             },
    { "Polygon"       , DataTypes::POLYGON          },
    { "Feature"       , DataTypes::FEATURE          },
    { "All"           , DataTypes::ALL              },
    { "DependsOnInput", DataTypes::DEPENDS_ON_INPUT }
};


static std::map<detail::DataType, String> string_by_data_type = {
    { detail::DataType::DT_SCALAR          , "Scalar"         },
    { detail::DataType::DT_POINT           , "Point"          },
    { detail::DataType::DT_LINE            , "Line"           },
    { detail::DataType::DT_POLYGON         , "Polygon"        },
    { detail::DataType::DT_DEPENDS_ON_INPUT, "DependsOnInput" }
};


DataTypes DataTypes::from_string(
    String const& string)
{
    assert(!string.is_empty());
    assert(data_type_by_string.find(string) != data_type_by_string.end());
    return data_type_by_string[string];
}


std::vector<detail::DataType> const DataTypes::DATA_TYPES = {
    detail::DataType::DT_SCALAR,
    detail::DataType::DT_POINT,
    detail::DataType::DT_LINE,
    detail::DataType::DT_POLYGON,
    detail::DataType::DT_DEPENDS_ON_INPUT
};


static String to_string(
    detail::DataType data_type)
{
    assert(string_by_data_type.find(data_type) != string_by_data_type.end());
    return string_by_data_type[data_type];
}


DataTypes::DataTypes()

    : FlagCollection<DataTypes, detail::DataType,
        detail::DataType::DT_NR_DATA_TYPES>()

{
}


DataTypes::DataTypes(
    std::set<detail::DataType> const& data_types)

    : FlagCollection<DataTypes, detail::DataType,
        detail::DataType::DT_NR_DATA_TYPES>(data_types)

{
}


String DataTypes::to_string() const
{
    assert(DataTypes::DATA_TYPES.size() == detail::DT_NR_DATA_TYPES);
    std::vector<String> strings;

    for(detail::DataType data_type: DataTypes::DATA_TYPES) {
        if(test(data_type)) {
            strings.push_back(ranally::to_string(data_type));
        }
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
