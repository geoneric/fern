#include "fern/python_extension/algorithm/numpy/sqrt.h"
#include <functional>
#include <map>
#include "fern/core/types.h"
#include "fern/feature/core/array_reference_traits.h"
#include "fern/algorithm/algebra/elementary/sqrt.h"
#include "fern/python_extension/core/error.h"
#include "fern/python_extension/algorithm/core/macro.h"
#include "fern/python_extension/algorithm/numpy/numpy_type_traits.h"
#include "fern/python_extension/algorithm/numpy/util.h"


namespace fern {
namespace python {
namespace detail {

static void init_numpy()
{
    import_array();
}


namespace array {

template<
    typename Value>
PyArrayObject* sqrt(
    PyArrayObject* array_object)
{
    init_numpy();

    // TODO Switch on number of dimensions.
    assert(PyArray_NDIM(array_object) == 2);

    size_t const size1{static_cast<size_t>(PyArray_DIM(array_object, 0))};
    size_t const size2{static_cast<size_t>(PyArray_DIM(array_object, 1))};

    ArrayReference<Value, 2> array_2d_reference(
        static_cast<Value*>(PyArray_DATA(array_object)), extents[size1][size2]);

    using result_value_type = Value;

    PyArrayObject* result_object{(PyArrayObject*)(
        PyArray_SimpleNew(
            PyArray_NDIM(array_object),
            PyArray_DIMS(array_object),
            NumpyTypeTraits<result_value_type>::data_type))};

    ArrayReference<result_value_type, 2> result_reference(
        static_cast<result_value_type*>(PyArray_DATA(result_object)),
        extents[size1][size2]);

    algorithm::algebra::sqrt(algorithm::parallel, array_2d_reference,
        result_reference);

    return result_object;
}


#define UFO UNARY_FUNCTION_OVERLOAD

UFO(sqrt, PyArrayObject*, float32)
UFO(sqrt, PyArrayObject*, float64)

#undef UFO


using SqrtOverloadsKey = int;
using SqrtOverload = std::function<PyArrayObject*(PyArrayObject*)>;
using SqrtOverloads = std::map<SqrtOverloadsKey, SqrtOverload>;


static SqrtOverloads add_overloads = {
    { SqrtOverloadsKey(NPY_FLOAT32), sqrt_float32 },
    { SqrtOverloadsKey(NPY_FLOAT64), sqrt_float64 }
};

} // namespace array
} // namespace detail


PyArrayObject* sqrt(
    PyArrayObject* array)
{
    using namespace detail::array;

    int data_type = PyArray_TYPE(array);
    SqrtOverloadsKey key(data_type);

    PyArrayObject* result{nullptr};

    if(add_overloads.find(key) == add_overloads.end()) {
        raise_unsupported_overload_exception(python_object(array));
        result = nullptr;
    }
    else {
        result = add_overloads[key](array);
    }

    return result;
}

} // namespace python
} // namespace fern
