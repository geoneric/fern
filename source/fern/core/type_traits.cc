// -----------------------------------------------------------------------------
// Fern © Geoneric
//
// This file is part of Geoneric Fern which is available under the terms of
// the GNU General Public License (GPL), version 2. If you do not want to
// be bound by the terms of the GPL, you may purchase a proprietary license
// from Geoneric (http://www.geoneric.eu/contact).
// -----------------------------------------------------------------------------
#include "fern/core/type_traits.h"


namespace fern {

// These are defined here because we have to be absolutely sure that the
// are defined before the trait members are defined. See also the remark in
// value_types.cc
ValueTypes const ValueTypes::UNKNOWN;
ValueTypes const ValueTypes::BOOL(1 << ValueType::VT_BOOL);
ValueTypes const ValueTypes::CHAR(1 << ValueType::VT_CHAR);
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
ValueTypes const ValueTypes::ALL(ValueTypes::BOOL | ValueTypes::CHAR |
    ValueTypes::NUMBER | ValueTypes::STRING);

ValueType const TypeTraits<bool>::value_type(VT_BOOL);
ValueTypes const TypeTraits<bool>::value_types(ValueTypes::BOOL);
std::string const TypeTraits<bool>::name("bool");

ValueType const TypeTraits<char>::value_type(VT_CHAR);
ValueTypes const TypeTraits<char>::value_types(ValueTypes::CHAR);
std::string const TypeTraits<char>::name("char");

ValueType const TypeTraits<int8_t>::value_type(VT_INT8);
ValueTypes const TypeTraits<int8_t>::value_types(ValueTypes::INT8);
std::string const TypeTraits<int8_t>::name("int8");

ValueType const TypeTraits<uint8_t>::value_type(VT_UINT8);
ValueTypes const TypeTraits<uint8_t>::value_types(ValueTypes::UINT8);
std::string const TypeTraits<uint8_t>::name("uint8");

ValueType const TypeTraits<int16_t>::value_type(VT_INT16);
ValueTypes const TypeTraits<int16_t>::value_types(ValueTypes::INT16);
std::string const TypeTraits<int16_t>::name("int16");

ValueType const TypeTraits<uint16_t>::value_type(VT_UINT16);
ValueTypes const TypeTraits<uint16_t>::value_types(ValueTypes::UINT16);
std::string const TypeTraits<uint16_t>::name("uint16");

ValueType const TypeTraits<int32_t>::value_type(VT_INT32);
ValueTypes const TypeTraits<int32_t>::value_types(ValueTypes::INT32);
std::string const TypeTraits<int32_t>::name("int32");

ValueType const TypeTraits<uint32_t>::value_type(VT_UINT32);
ValueTypes const TypeTraits<uint32_t>::value_types(ValueTypes::UINT32);
std::string const TypeTraits<uint32_t>::name("uint32");

ValueType const TypeTraits<int64_t>::value_type(VT_INT64);
ValueTypes const TypeTraits<int64_t>::value_types(ValueTypes::INT64);
std::string const TypeTraits<int64_t>::name("int64");

ValueType const TypeTraits<uint64_t>::value_type(VT_UINT64);
ValueTypes const TypeTraits<uint64_t>::value_types(ValueTypes::UINT64);
std::string const TypeTraits<uint64_t>::name("uint64");

ValueType const TypeTraits<float>::value_type(VT_FLOAT32);
ValueTypes const TypeTraits<float>::value_types(ValueTypes::FLOAT32);
std::string const TypeTraits<float>::name("float32");

ValueType const TypeTraits<double>::value_type(VT_FLOAT64);
ValueTypes const TypeTraits<double>::value_types(ValueTypes::FLOAT64);
std::string const TypeTraits<double>::name("float64");

ValueType const TypeTraits<std::string>::value_type(VT_STRING);
ValueTypes const TypeTraits<std::string>::value_types(ValueTypes::STRING);
std::string const TypeTraits<std::string>::name("string");

}
