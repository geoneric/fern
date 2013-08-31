#include "geoneric/operation/core/type_traits.h"


namespace geoneric {

ValueType const TypeTraits<int8_t>::value_type(VT_INT8);
ValueTypes const TypeTraits<int8_t>::value_types(ValueTypes::INT8);

ValueType const TypeTraits<uint8_t>::value_type(VT_UINT8);
ValueTypes const TypeTraits<uint8_t>::value_types(ValueTypes::UINT8);

ValueType const TypeTraits<int16_t>::value_type(VT_INT16);
ValueTypes const TypeTraits<int16_t>::value_types(ValueTypes::INT16);

ValueType const TypeTraits<uint16_t>::value_type(VT_UINT16);
ValueTypes const TypeTraits<uint16_t>::value_types(ValueTypes::UINT16);

ValueType const TypeTraits<int32_t>::value_type(VT_INT32);
ValueTypes const TypeTraits<int32_t>::value_types(ValueTypes::INT32);

ValueType const TypeTraits<uint32_t>::value_type(VT_UINT32);
ValueTypes const TypeTraits<uint32_t>::value_types(ValueTypes::UINT32);

ValueType const TypeTraits<int64_t>::value_type(VT_INT64);
ValueTypes const TypeTraits<int64_t>::value_types(ValueTypes::INT64);

ValueType const TypeTraits<uint64_t>::value_type(VT_UINT64);
ValueTypes const TypeTraits<uint64_t>::value_types(ValueTypes::UINT64);

ValueType const TypeTraits<float>::value_type(VT_FLOAT32);
ValueTypes const TypeTraits<float>::value_types(ValueTypes::FLOAT32);

ValueType const TypeTraits<double>::value_type(VT_FLOAT64);
ValueTypes const TypeTraits<double>::value_types(ValueTypes::FLOAT64);

ValueType const TypeTraits<String>::value_type(VT_STRING);
ValueTypes const TypeTraits<String>::value_types(ValueTypes::STRING);

}
