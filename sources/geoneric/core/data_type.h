#pragma once


namespace geoneric {

enum DataType {

    //! A single value, constant in time and space.
    DT_CONSTANT,

    //! A 2D field of spatially varying values, constant in time.
    DT_STATIC_FIELD,

    // DT_POINT,

    // DT_LINE,

    // DT_POLYGON,

    DT_LAST_DATA_TYPE = DT_STATIC_FIELD

};

} // namespace geoneric
