#include "fern/python_extension/feature/masked_raster.h"
#include <iostream>
#include <memory>
#include "fern/feature/core/array_traits.h"
#include "fern/feature/core/masked_raster.h"


namespace fern {
namespace python {
template<
    typename T>
static void copy(
    boost::python::list& values,
    Array<T, 2>& array)
{
    namespace bp = boost::python;

    for(size_t row = 0; row < size(array, 0); ++row) {
        bp::list row_object(values[row]);

        for(size_t col = 0; col < size(array, 1); ++col) {
            bp::object value_object(row_object[col]);
            get(array, row, col) = bp::extract<T>(value_object);
        }
    }
}


#define HANDLE_CASE(                                                 \
    value_type_enum, value_type)                                     \
case value_type_enum: {                                              \
    using MaskedRaster = fern::MaskedRaster<value_type, 2>;          \
    using Transformation = MaskedRaster::Transformation;             \
    Transformation transformation{{_origin.first, _cell_sizes.first, \
        _origin.second, _cell_sizes.second}};                        \
    _masked_raster = std::make_shared<MaskedRaster>(                 \
        fern::extents[_sizes.first][_sizes.second], transformation); \
    break;                                                           \
}

MaskedRaster::MaskedRaster(
    boost::python::tuple const& sizes,
    boost::python::tuple const& origin,
    boost::python::tuple const& cell_sizes,
    fern::ValueType value_type)

    : _sizes(boost::python::extract<size_t>(sizes[0])(),
          boost::python::extract<size_t>(sizes[1])()),
      _origin(boost::python::extract<double>(origin[0])(),
          boost::python::extract<double>(origin[1])()),
      _cell_sizes(boost::python::extract<double>(cell_sizes[0])(),
          boost::python::extract<double>(cell_sizes[1])()),
      _value_type(value_type)

{
    switch(value_type) {
        HANDLE_CASE(VT_UINT8, uint8_t)
        HANDLE_CASE(VT_INT8, int8_t)
        HANDLE_CASE(VT_UINT16, uint16_t)
        HANDLE_CASE(VT_INT16, int16_t)
        HANDLE_CASE(VT_UINT32, uint32_t)
        HANDLE_CASE(VT_INT32, int32_t)
        HANDLE_CASE(VT_UINT64, uint64_t)
        HANDLE_CASE(VT_INT64, int64_t)
        HANDLE_CASE(VT_FLOAT32, float)
        HANDLE_CASE(VT_FLOAT64, double)
        case VT_BOOL:
        case VT_STRING: {
            // These value types are not exposed in Python, so they shouldn't
            // arrive here.
            assert(false);
        }
    }

    assert(!_masked_raster.empty());
}

#undef HANDLE_CASE


#define HANDLE_CASE(                                                 \
    value_type_enum, value_type)                                     \
case value_type_enum: {                                              \
    using MaskedRaster = fern::MaskedRaster<value_type, 2>;          \
    using Transformation = MaskedRaster::Transformation;             \
    Transformation transformation{{_origin.first, _cell_sizes.first, \
        _origin.second, _cell_sizes.second}};                        \
    auto masked_raster = std::make_shared<MaskedRaster>(             \
        fern::extents[_sizes.first][_sizes.second], transformation); \
    _masked_raster = masked_raster;                                  \
    copy(values, *masked_raster);                                    \
    copy(mask, masked_raster->mask());                               \
    break;                                                           \
}

MaskedRaster::MaskedRaster(
    boost::python::list& values,
    boost::python::list& mask,
    boost::python::tuple& origin,
    boost::python::tuple& cell_sizes,
    fern::ValueType value_type)

    : _sizes(
          boost::python::extract<size_t>(values.attr("__len__")()),
          boost::python::extract<size_t>(values[0].attr("__len__")())),
      _origin(boost::python::extract<double>(origin[0])(),
          boost::python::extract<double>(origin[1])()),
      _cell_sizes(boost::python::extract<double>(cell_sizes[0])(),
          boost::python::extract<double>(cell_sizes[1])()),
      _value_type(value_type)

{
    namespace bp = boost::python;

    switch(value_type) {
        HANDLE_CASE(VT_UINT8, uint8_t)
        HANDLE_CASE(VT_INT8, int8_t)
        HANDLE_CASE(VT_UINT16, uint16_t)
        HANDLE_CASE(VT_INT16, int16_t)
        HANDLE_CASE(VT_UINT32, uint32_t)
        HANDLE_CASE(VT_INT32, int32_t)
        HANDLE_CASE(VT_UINT64, uint64_t)
        HANDLE_CASE(VT_INT64, int64_t)
        HANDLE_CASE(VT_FLOAT32, float)
        HANDLE_CASE(VT_FLOAT64, double)
        case VT_BOOL:
        case VT_STRING: {
            // These value types are not exposed in Python, so they shouldn't
            // arrive here.
            assert(false);
        }
    }

    assert(!_masked_raster.empty());
}

#undef HANDLE_CASE


boost::python::tuple MaskedRaster::sizes() const
{
    return boost::python::make_tuple(_sizes.first, _sizes.second);
}


boost::python::tuple MaskedRaster::origin() const
{
    return boost::python::make_tuple(_origin.first, _origin.second);
}


boost::python::tuple MaskedRaster::cell_sizes() const
{
    return boost::python::make_tuple(_cell_sizes.first, _cell_sizes.second);
}

} // namespace python
} // namespace fern
