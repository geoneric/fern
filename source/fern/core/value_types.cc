// -----------------------------------------------------------------------------
// Fern © Geoneric
//
// This file is part of Geoneric Fern which is available under the terms of
// the GNU General Public License (GPL), version 2. If you do not want to
// be bound by the terms of the GPL, you may purchase a proprietary license
// from Geoneric (http://www.geoneric.eu/contact).
// -----------------------------------------------------------------------------
#include "fern/core/value_types.h"
#include "fern/core/string.h"
#include <cassert>
#include <map>


namespace fern {

// The static ValueTypes members are defined in type_traits.cc. Otherwise,
// if type_traits.cc is built before this module, it won't have the correct
// values yet (the value_types member will all be ValueTypes::UNKNOWN).

// These strings should match the ones used in the XML schema.
static std::map<std::string, ValueTypes> value_type_by_string = {
    { "Bool"          , ValueTypes::BOOL             },
    { "Char"          , ValueTypes::CHAR             },
    { "Uint8"         , ValueTypes::UINT8            },
    { "Int8"          , ValueTypes::INT8             },
    { "Uint16"        , ValueTypes::UINT16           },
    { "Int16"         , ValueTypes::INT16            },
    { "Uint32"        , ValueTypes::UINT32           },
    { "Int32"         , ValueTypes::INT32            },
    { "Uint64"        , ValueTypes::UINT64           },
    { "Int64"         , ValueTypes::INT64            },
    { "Size"          , ValueTypes::SIZE             },
    { "Float32"       , ValueTypes::FLOAT32          },
    { "Float64"       , ValueTypes::FLOAT64          },
    { "String"        , ValueTypes::STRING           },
    { "Number"        , ValueTypes::NUMBER           },
    { "All"           , ValueTypes::ALL              }
};


static std::map<ValueType, std::string> string_by_value_type = {
    { ValueType::VT_BOOL            , "Bool"           },
    { ValueType::VT_CHAR            , "Char"           },
    { ValueType::VT_UINT8           , "Uint8"          },
    { ValueType::VT_INT8            , "Int8"           },
    { ValueType::VT_UINT16          , "Uint16"         },
    { ValueType::VT_INT16           , "Int16"          },
    { ValueType::VT_UINT32          , "Uint32"         },
    { ValueType::VT_INT32           , "Int32"          },
    { ValueType::VT_UINT64          , "Uint64"         },
    { ValueType::VT_INT64           , "Int64"          },
    { ValueType::VT_FLOAT32         , "Float32"        },
    { ValueType::VT_FLOAT64         , "Float64"        },
    { ValueType::VT_STRING          , "String"         }
};


static std::string to_string(
    ValueType value_type)
{
    assert(string_by_value_type.find(value_type) != string_by_value_type.end());
    return string_by_value_type[value_type];
}


std::vector<ValueType> const ValueTypes::VALUE_TYPES = {
    ValueType::VT_BOOL,
    ValueType::VT_CHAR,
    ValueType::VT_UINT8,
    ValueType::VT_INT8,
    ValueType::VT_UINT16,
    ValueType::VT_INT16,
    ValueType::VT_UINT32,
    ValueType::VT_INT32,
    ValueType::VT_UINT64,
    ValueType::VT_INT64,
    ValueType::VT_FLOAT32,
    ValueType::VT_FLOAT64,
    ValueType::VT_STRING
};


ValueTypes ValueTypes::from_string(
    std::string const& string)
{
    assert(!string.empty());
    assert(value_type_by_string.find(string) != value_type_by_string.end());
    return value_type_by_string[string];
}


ValueTypes::ValueTypes()

    : FlagCollection<ValueTypes, ValueType,
          ValueType::VT_LAST_VALUE_TYPE + 1>()

{
}


std::string ValueTypes::to_string() const
{
    assert(ValueTypes::VALUE_TYPES.size() == VT_LAST_VALUE_TYPE + 1);
    std::vector<std::string> strings;

    for(ValueType value_type: ValueTypes::VALUE_TYPES) {
        if(test(value_type)) {
            strings.emplace_back(fern::to_string(value_type));
        }
    }

    if(strings.empty()) {
        strings.emplace_back("?");
    }

    return join(strings, "|");
}


ValueTypes operator|(
    ValueTypes const& lhs,
    ValueTypes const& rhs)
{
    ValueTypes result(lhs);
    result |= rhs;
    return result;
}


std::ostream& operator<<(
    std::ostream& stream,
    ValueTypes const& flags)
{
    stream << flags.to_string();
    return stream;
}

} // namespace fern
