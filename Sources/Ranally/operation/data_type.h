#pragma once


namespace ranally {

enum DataType {

    //! Unknown data type.
    DT_UNKNOWN=0,

    //! Single value.
    DT_VALUE=1,

    //! Raster with values.
    DT_RASTER=2,

    //! Feature layer with values.
    DT_FEATURE=4,

    //! Spatial data type.
    DT_SPATIAL=DT_RASTER | DT_FEATURE,

    //! All data types.
    DT_ALL=DT_VALUE | DT_RASTER | DT_FEATURE,

    //! Data type depends on data type of input.
    DT_DEPENDS_ON_INPUT=8

};

typedef unsigned int DataTypes;

} // namespace ranally
