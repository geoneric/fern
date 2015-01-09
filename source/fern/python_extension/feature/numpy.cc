#include "fern/python_extension/feature/masked_raster.h"
#include <numpy/ndarrayobject.h>
#include "fern/python_extension/core/switch_on_value_type.h"
#include "fern/python_extension/feature/numpy.h"
#include "fern/python_extension/algorithm/tmp/numpy/numpy_type_traits.h"


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
    detail::MaskedRaster<T> const& masked_raster)
{
    int const nr_dimensions{2};

    assert(std::get<0>(masked_raster.sizes()) < static_cast<size_t>(
        std::numeric_limits<npy_intp>::max()));
    assert(std::get<1>(masked_raster.sizes()) < static_cast<size_t>(
        std::numeric_limits<npy_intp>::max()));
    npy_intp dimensions[2] = {
        static_cast<npy_intp>(std::get<0>(masked_raster.sizes())),
        static_cast<npy_intp>(std::get<1>(masked_raster.sizes())) };

    T* data = const_cast<T*>(masked_raster.data());

    bp::numeric::array array_object(bp::object(bp::handle<>(
        PyArray_SimpleNewFromData(nr_dimensions, dimensions,
            NumpyTypeTraits<T>::data_type, data))));

    return bp::object(bp::handle<>(
        PyArray_SimpleNewFromData(nr_dimensions, dimensions,
            NumpyTypeTraits<T>::data_type, data)));
}

} // anonymous namespace


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


// #define CASE(                                                     
//     value_type_enum,                                              
//     value_type)                                                   
// case value_type_enum: {                                           
//     auto const& array(masked_raster.raster<value_type>().mask()); 
//     object = wrap_array(array);                                   
//     break;                                                        
// }
// 
// bp::object mask_as_numpy_array(
//     fp::MaskedRaster const& masked_raster)
// {
//     assert(!PyErr_Occurred());
//     init_numpy();
//     bp::object object;
// 
//     SWITCH_ON_VALUE_TYPE(masked_raster.value_type(), CASE)
// 
//     assert(!PyErr_Occurred());
//     return bp::numeric::array{object}.copy();
// }
// 
// #undef CASE

} // namespace python
} // namespace fern
