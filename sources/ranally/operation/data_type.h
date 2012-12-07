#pragma once


namespace ranally {

enum DataType {

    // //! Unknown data type.
    // DT_UNKNOWN=0,

    // //! Single value.
    // DT_VALUE=1,

    // //! Raster with values.
    // DT_RASTER=2,

    // //! Feature layer with values.
    // DT_FEATURE=4,

    // //! Spatial data type.
    // DT_SPATIAL=DT_RASTER | DT_FEATURE,

    // //! All data types.
    // DT_ALL=DT_VALUE | DT_RASTER | DT_FEATURE,

    // //! Data type depends on data type of input.
    // DT_DEPENDS_ON_INPUT=8

    //! Unknown data type.
    DT_UNKNOWN=0,

    DT_SCALAR=1,

    DT_POINT=2,

    DT_LINE=4,

    DT_POLYGON=8,

    //! Data type depends on data type of input.
    DT_DEPENDS_ON_INPUT=16,

    //! Shortcut for all feature types.
    DT_FEATURE=DT_POINT | DT_LINE | DT_POLYGON,

    //! Shortcut for all data types.
    DT_ALL=DT_SCALAR | DT_FEATURE,

};

typedef unsigned int DataTypes;

} // namespace ranally
