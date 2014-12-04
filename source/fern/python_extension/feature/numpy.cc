#include "fern/python_extension/feature/masked_raster.h"
#include <iostream>
#include <numpy/ndarrayobject.h>
#include "fern/python_extension/algorithm/numpy/numpy_type_traits.h"
#include "fern/python_extension/feature/numpy.h"


namespace bp = boost::python;
namespace fp = fern::python;

namespace fern {
namespace python {
namespace {

static void init_numpy()
{
    import_array();
}


template<
    typename T>
bp::object wrap_array(
    Array<T, 2> const& array)
{
    int const nr_dimensions{2};

    assert(array.shape()[0] < static_cast<size_t>(
        std::numeric_limits<npy_intp>::max()));
    assert(array.shape()[1] < static_cast<size_t>(
        std::numeric_limits<npy_intp>::min()));
    npy_intp dimensions[2] = {
        static_cast<npy_intp>(array.shape()[0]),
        static_cast<npy_intp>(array.shape()[1]) };

    T* data = const_cast<T*>(array.data());

    bp::numeric::array array_object(bp::object(bp::handle<>(
        PyArray_SimpleNewFromData(nr_dimensions, dimensions,
            NumpyTypeTraits<T>::data_type, data))));

    return bp::object(bp::handle<>(
        PyArray_SimpleNewFromData(nr_dimensions, dimensions,
            NumpyTypeTraits<T>::data_type, data)));
}

} // anonymous namespace


// VT_BOOL and VT_STRING are not exposed in Python, so they shouldn't
// be passed in.
#define SWITCH_ON_VALUE_TYPE(   \
    value_type,                 \
    case_)                      \
switch(value_type) {            \
    case_(VT_UINT8, uint8_t)    \
    case_(VT_INT8, int8_t)      \
    case_(VT_UINT16, uint16_t)  \
    case_(VT_INT16, int16_t)    \
    case_(VT_UINT32, uint32_t)  \
    case_(VT_INT32, int32_t)    \
    case_(VT_UINT64, uint64_t)  \
    case_(VT_INT64, int64_t)    \
    case_(VT_FLOAT32, float)    \
    case_(VT_FLOAT64, double)   \
    case VT_BOOL:               \
    case VT_STRING: {           \
        assert(false);          \
    }                           \
}


#define CASE(                                              \
    value_type_enum,                                       \
    value_type)                                            \
case value_type_enum: {                                    \
    auto const& array(masked_raster.raster<value_type>()); \
    object = wrap_array(array);                            \
    break;                                                 \
}

bp::object raster_as_numpy_array(
    fp::MaskedRaster const& masked_raster)
{
    assert(!PyErr_Occurred());
    init_numpy();
    bp::object object;

    SWITCH_ON_VALUE_TYPE(masked_raster.value_type(), CASE)

    assert(!PyErr_Occurred());
    return bp::numeric::array{object}.copy();
}

#undef CASE


#define CASE(                                                     \
    value_type_enum,                                              \
    value_type)                                                   \
case value_type_enum: {                                           \
    auto const& array(masked_raster.raster<value_type>().mask()); \
    object = wrap_array(array);                                   \
    break;                                                        \
}

bp::object mask_as_numpy_array(
    fp::MaskedRaster const& masked_raster)
{
    assert(!PyErr_Occurred());
    init_numpy();
    bp::object object;

    SWITCH_ON_VALUE_TYPE(masked_raster.value_type(), CASE)

    assert(!PyErr_Occurred());
    return bp::numeric::array{object}.copy();
}

#undef CASE

} // namespace python
} // namespace fern
