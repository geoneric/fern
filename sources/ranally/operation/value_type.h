#pragma once


namespace ranally {
namespace detail {

enum ValueType {

    VT_UINT8,

    VT_INT8,

    VT_UINT16,

    VT_INT16,

    VT_UINT32,

    VT_INT32,

    VT_UINT64,

    VT_INT64,

    VT_FLOAT32,

    VT_FLOAT64,

    //! String value.
    VT_STRING,

    VT_NR_VALUE_TYPES
};

} // namespace detail
} // namespace ranally
