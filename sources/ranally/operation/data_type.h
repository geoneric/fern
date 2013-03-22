#pragma once


namespace ranally {
namespace detail {

enum DataType {

    DT_SCALAR,

    DT_POINT,

    DT_LINE,

    DT_POLYGON,

    // //! Data type depends on data type of input.
    // DT_DEPENDS_ON_INPUT,

    DT_NR_DATA_TYPES
};

} // namespace detail
} // namespace ranally
