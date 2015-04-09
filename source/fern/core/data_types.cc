// -----------------------------------------------------------------------------
// Fern © Geoneric
//
// This file is part of Geoneric Fern which is available under the terms of
// the GNU General Public License (GPL), version 2. If you do not want to
// be bound by the terms of the GPL, you may purchase a proprietary license
// from Geoneric (http://www.geoneric.eu/contact).
// -----------------------------------------------------------------------------
#include "fern/core/data_types.h"
#include <cassert>
#include <map>


namespace fern {

DataTypes const DataTypes::UNKNOWN;
DataTypes const DataTypes::CONSTANT(1 << DataType::DT_CONSTANT);
DataTypes const DataTypes::STATIC_FIELD(1 << DataType::DT_STATIC_FIELD);
// DataTypes const DataTypes::POINT(1 << DataType::DT_POINT);
// DataTypes const DataTypes::LINE(1 << DataType::DT_LINE);
// DataTypes const DataTypes::POLYGON(1 << DataType::DT_POLYGON);
// DataTypes const DataTypes::FEATURE(DataTypes::POINT | DataTypes::LINE |
//     DataTypes::POLYGON);
DataTypes const DataTypes::ALL(DataTypes::CONSTANT | DataTypes::STATIC_FIELD);


// These strings should match the ones used in the XML schema.
static std::map<String, DataTypes> data_type_by_string = {
    { "Constant", DataTypes::CONSTANT },
    { "StaticField", DataTypes::STATIC_FIELD },
    // { "Point", DataTypes::POINT },
    // { "Line", DataTypes::LINE },
    // { "Polygon", DataTypes::POLYGON },
    // { "Feature", DataTypes::FEATURE },
    { "All"     , DataTypes::ALL }
};


static std::map<DataType, String> string_by_data_type = {
    { DataType::DT_CONSTANT, "Constant" },
    { DataType::DT_STATIC_FIELD, "StaticField" }
    // { DataType::DT_POINT   , "Point"    },
    // { DataType::DT_LINE    , "Line"     },
    // { DataType::DT_POLYGON , "Polygon"  }
};


static String to_string(
    DataType data_type)
{
    assert(string_by_data_type.find(data_type) != string_by_data_type.end());
    return string_by_data_type[data_type];
}


DataTypes DataTypes::from_string(
    String const& string)
{
    assert(!string.is_empty());
    assert(data_type_by_string.find(string) != data_type_by_string.end());
    return data_type_by_string[string];
}


std::vector<DataType> const DataTypes::DATA_TYPES = {
    DataType::DT_CONSTANT,
    DataType::DT_STATIC_FIELD
    // DataType::DT_POINT,
    // DataType::DT_LINE,
    // DataType::DT_POLYGON
};


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
            strings.emplace_back(fern::to_string(data_type));
        }
    }

    if(strings.empty()) {
        strings.emplace_back("?");
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

} // namespace fern
