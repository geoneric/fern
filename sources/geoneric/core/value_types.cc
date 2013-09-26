#include "geoneric/core/value_types.h"
#include <map>


namespace geoneric {

ValueTypes const ValueTypes::UNKNOWN;
ValueTypes const ValueTypes::UINT8(1 << ValueType::VT_UINT8);
ValueTypes const ValueTypes::INT8(1 << ValueType::VT_INT8);
ValueTypes const ValueTypes::UINT16(1 << ValueType::VT_UINT16);
ValueTypes const ValueTypes::INT16(1 << ValueType::VT_INT16);
ValueTypes const ValueTypes::UINT32(1 << ValueType::VT_UINT32);
ValueTypes const ValueTypes::INT32(1 << ValueType::VT_INT32);
ValueTypes const ValueTypes::UINT64(1 << ValueType::VT_UINT64);
ValueTypes const ValueTypes::INT64(1 << ValueType::VT_INT64);
ValueTypes const ValueTypes::FLOAT32(1 << ValueType::VT_FLOAT32);
ValueTypes const ValueTypes::FLOAT64(1 << ValueType::VT_FLOAT64);
ValueTypes const ValueTypes::STRING(1 << ValueType::VT_STRING);
ValueTypes const ValueTypes::UNSIGNED_INTEGER(ValueTypes::UINT8 |
    ValueTypes::UINT16 | ValueTypes::UINT32 | ValueTypes::UINT64);
ValueTypes const ValueTypes::SIGNED_INTEGER(ValueTypes::INT8 |
    ValueTypes::INT16 | ValueTypes::INT32 | ValueTypes::INT64);
ValueTypes const ValueTypes::INTEGER(ValueTypes::UNSIGNED_INTEGER |
    ValueTypes::SIGNED_INTEGER);
ValueTypes const ValueTypes::SIZE(ValueTypes::UINT64);
ValueTypes const ValueTypes::FLOATING_POINT(ValueTypes::FLOAT32 |
    ValueTypes::FLOAT64);
ValueTypes const ValueTypes::NUMBER(ValueTypes::INTEGER |
    ValueTypes::FLOATING_POINT);
ValueTypes const ValueTypes::ALL(ValueTypes::NUMBER | ValueTypes::STRING);

// These strings should match the ones used in the XML schema.
static std::map<String, ValueTypes> value_type_by_string = {
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


static std::map<ValueType, String> string_by_value_type = {
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


static String to_string(
    ValueType value_type)
{
    assert(string_by_value_type.find(value_type) != string_by_value_type.end());
    return string_by_value_type[value_type];
}


std::vector<ValueType> const ValueTypes::VALUE_TYPES = {
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
    String const& string)
{
    assert(!string.is_empty());
    assert(value_type_by_string.find(string) != value_type_by_string.end());
    return value_type_by_string[string];
}


ValueTypes::ValueTypes()

    : FlagCollection<ValueTypes, ValueType,
          ValueType::VT_LAST_VALUE_TYPE + 1>()

{
}


String ValueTypes::to_string() const
{
    assert(ValueTypes::VALUE_TYPES.size() == VT_LAST_VALUE_TYPE + 1);
    std::vector<String> strings;

    for(ValueType value_type: ValueTypes::VALUE_TYPES) {
        if(test(value_type)) {
            strings.push_back(geoneric::to_string(value_type));
        }
    }

    if(strings.empty()) {
        strings.push_back("?");
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

} // namespace geoneric
