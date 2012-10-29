#pragma once


namespace ranally {
namespace operation {

enum ValueType {
  //! Unknown value type.
  VT_UNKNOWN=0,

  VT_UINT8=1,
  VT_INT8=2,
  VT_UINT16=4,
  VT_INT16=8,
  VT_UINT32=16,
  VT_INT32=32,
  VT_UINT64=64,
  VT_INT64=128,
  VT_FLOAT32=256,
  VT_FLOAT64=512,

  //! String value.
  VT_STRING=1024,

  //! Unsigned integer value type.
  VT_UNSIGNED_INTEGER=VT_UINT8 | VT_UINT16 | VT_UINT32 | VT_UINT64,

  //! Signed integer value type.
  VT_SIGNED_INTEGER=VT_INT8 | VT_INT16 | VT_INT32 | VT_INT64,

  //! Integral value type.
  VT_INTEGER=VT_UNSIGNED_INTEGER | VT_SIGNED_INTEGER,

  //! Floatint point value type.
  VT_FLOATING_POINT=VT_FLOAT32 | VT_FLOAT64,

  //! Numeric value type.
  VT_NUMBER=VT_INTEGER | VT_FLOATING_POINT,

  //! All value types.
  VT_ALL=VT_NUMBER | VT_STRING,

  //! Value type depends on value type of input.
  VT_DEPENDS_ON_INPUT=2048
};

typedef unsigned int ValueTypes;

} // namespace operation
} // namespace ranally
