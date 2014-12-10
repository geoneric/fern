#include "fern/python_extension/algorithm/numpy/add.h"
#include <functional>
#include <map>
#include <tuple>
#include "fern/feature/core/array_reference_traits.h"
#include "fern/algorithm/algebra/elementary/add.h"
#include "fern/python_extension/core/error.h"
#include "fern/python_extension/algorithm/core/macro.h"
#include "fern/python_extension/algorithm/core/util.h"
#include "fern/python_extension/algorithm/numpy/numpy_type_traits.h"
#include "fern/python_extension/algorithm/numpy/util.h"


namespace fern {
namespace python {
namespace numpy {
namespace detail {

static void init_numpy()
{
    import_array();
}


namespace array_number {

template<
    typename Value1,
    typename Value2>
PyArrayObject* add(
    PyArrayObject* array_object,
    Value2 const& value)
{
    assert(!PyErr_Occurred());

    init_numpy();

    // TODO Verify array is contiguous and aligned.
    size_t const size{PyArray_SIZE(array_object)};

    ArrayReference<Value1, 1> array_reference(static_cast<Value1*>(
            PyArray_DATA(array_object)), extents[size]);

    using result_value_type = algorithm::add::result_value_type<Value1, Value2>;

    PyArrayObject* result_object{(PyArrayObject*)(
        PyArray_SimpleNew(
            PyArray_NDIM(array_object),
            PyArray_DIMS(array_object),
            NumpyTypeTraits<result_value_type>::data_type))};
    assert(!PyErr_Occurred());

    ArrayReference<result_value_type, 1> result_reference(
        static_cast<result_value_type*>(PyArray_DATA(result_object)),
        extents[size]);

    algorithm::algebra::add(algorithm::parallel, array_reference,
        value, result_reference);

    assert(!PyErr_Occurred());

    return result_object;
}


#define BFO BINARY_FUNCTION_OVERLOAD

#define ADD_OVERLOADS(                                           \
    algorithm,                                                   \
    type)                                                        \
BFO(algorithm, PyArrayObject*, type, uint8_t const&, uint8)      \
BFO(algorithm, PyArrayObject*, type, int8_t const&, int8)        \
BFO(algorithm, PyArrayObject*, type, uint16_t const&, uint16)    \
BFO(algorithm, PyArrayObject*, type, int16_t const&, int16)      \
BFO(algorithm, PyArrayObject*, type, uint32_t const&, uint32)    \
BFO(algorithm, PyArrayObject*, type, int32_t const&, int32)      \
BFO(algorithm, PyArrayObject*, type, uint64_t const&, uint64)    \
BFO(algorithm, PyArrayObject*, type, int64_t const&, int64)      \
BFO(algorithm, PyArrayObject*, type, float32_t const&, float32)  \
BFO(algorithm, PyArrayObject*, type, float64_t const&, float64)

ADD_OVERLOADS(add, uint8)
ADD_OVERLOADS(add, int8)
ADD_OVERLOADS(add, uint16)
ADD_OVERLOADS(add, int16)
ADD_OVERLOADS(add, uint32)
ADD_OVERLOADS(add, int32)
ADD_OVERLOADS(add, uint64)
ADD_OVERLOADS(add, int64)
ADD_OVERLOADS(add, float32)
ADD_OVERLOADS(add, float64)

#undef ADD_OVERLOADS
#undef BFO

} // namespace array_number


namespace array_int64 {

using AddOverloadsKey = int;
using AddOverload = std::function<PyArrayObject*(PyArrayObject*, int64_t)>;
using AddOverloads = std::map<AddOverloadsKey, AddOverload>;


static AddOverloads add_overloads = {
    { AddOverloadsKey(NPY_UINT8   ), array_number::add_uint8_int64   },
    { AddOverloadsKey(NPY_INT8    ), array_number::add_int8_int64    },
    { AddOverloadsKey(NPY_UINT16  ), array_number::add_uint16_int64  },
    { AddOverloadsKey(NPY_INT16   ), array_number::add_int16_int64   },
    { AddOverloadsKey(NPY_UINT32  ), array_number::add_uint32_int64  },
    { AddOverloadsKey(NPY_INT32   ), array_number::add_int32_int64   },
    { AddOverloadsKey(NPY_UINT64  ), array_number::add_uint64_int64  },
    { AddOverloadsKey(NPY_INT64   ), array_number::add_int64_int64   },
    { AddOverloadsKey(NPY_FLOAT32 ), array_number::add_float32_int64 },
    { AddOverloadsKey(NPY_FLOAT64 ), array_number::add_float64_int64 }
};

} // namespace array_int64


namespace array_float64 {

using AddOverloadsKey = int;
using AddOverload = std::function<PyArrayObject*(PyArrayObject*, float64_t)>;
using AddOverloads = std::map<AddOverloadsKey, AddOverload>;


static AddOverloads add_overloads = {
    { AddOverloadsKey(NPY_UINT8   ), array_number::add_uint8_float64   },
    { AddOverloadsKey(NPY_INT8    ), array_number::add_int8_float64    },
    { AddOverloadsKey(NPY_UINT16  ), array_number::add_uint16_float64  },
    { AddOverloadsKey(NPY_INT16   ), array_number::add_int16_float64   },
    { AddOverloadsKey(NPY_UINT32  ), array_number::add_uint32_float64  },
    { AddOverloadsKey(NPY_INT32   ), array_number::add_int32_float64   },
    { AddOverloadsKey(NPY_UINT64  ), array_number::add_uint64_float64  },
    { AddOverloadsKey(NPY_INT64   ), array_number::add_int64_float64   },
    { AddOverloadsKey(NPY_FLOAT32 ), array_number::add_float32_float64 },
    { AddOverloadsKey(NPY_FLOAT64 ), array_number::add_float64_float64 }
};

} // namespace array_float64


namespace array_array {

template<
    typename Value1,
    typename Value2>
PyArrayObject* add(
    PyArrayObject* array_object1,
    PyArrayObject* array_object2)
{
    assert(!PyErr_Occurred());

    init_numpy();

    // TODO Raise exception.
    assert(PyArray_SIZE(array_object1) == PyArray_SIZE(array_object2));

    // TODO Verify array is contiguous and aligned.
    size_t const size{PyArray_SIZE(array_object1)};

    ArrayReference<Value1, 1> array_reference1(static_cast<Value1*>(
            PyArray_DATA(array_object1)), extents[size]);
    ArrayReference<Value2, 1> array_reference2(static_cast<Value2*>(
            PyArray_DATA(array_object2)), extents[size]);

    using result_value_type = algorithm::add::result_value_type<Value1, Value2>;

    PyArrayObject* result_object{(PyArrayObject*)(
        PyArray_SimpleNew(
            PyArray_NDIM(array_object1),
            PyArray_DIMS(array_object1),
            NumpyTypeTraits<result_value_type>::data_type))};
    assert(!PyErr_Occurred());

    ArrayReference<result_value_type, 1> result_reference(
        static_cast<result_value_type*>(PyArray_DATA(result_object)),
        extents[size]);

    algorithm::algebra::add(algorithm::parallel, array_reference1,
        array_reference2, result_reference);

    assert(!PyErr_Occurred());

    return result_object;
}


#define BFO BINARY_FUNCTION_OVERLOAD

#define ADD_OVERLOADS(                                         \
    algorithm,                                                 \
    type)                                                      \
BFO(algorithm, PyArrayObject*, type, PyArrayObject*, uint8)    \
BFO(algorithm, PyArrayObject*, type, PyArrayObject*, int8)     \
BFO(algorithm, PyArrayObject*, type, PyArrayObject*, uint16)   \
BFO(algorithm, PyArrayObject*, type, PyArrayObject*, int16)    \
BFO(algorithm, PyArrayObject*, type, PyArrayObject*, uint32)   \
BFO(algorithm, PyArrayObject*, type, PyArrayObject*, int32)    \
BFO(algorithm, PyArrayObject*, type, PyArrayObject*, uint64)   \
BFO(algorithm, PyArrayObject*, type, PyArrayObject*, int64)    \
BFO(algorithm, PyArrayObject*, type, PyArrayObject*, float32)  \
BFO(algorithm, PyArrayObject*, type, PyArrayObject*, float64)

ADD_OVERLOADS(add, uint8)
ADD_OVERLOADS(add, int8)
ADD_OVERLOADS(add, uint16)
ADD_OVERLOADS(add, int16)
ADD_OVERLOADS(add, uint32)
ADD_OVERLOADS(add, int32)
ADD_OVERLOADS(add, uint64)
ADD_OVERLOADS(add, int64)
ADD_OVERLOADS(add, float32)
ADD_OVERLOADS(add, float64)

#undef ADD_OVERLOADS
#undef BFO


using AddOverloadsKey = std::tuple<int, int>;
using AddOverload = std::function<PyArrayObject*(PyArrayObject*,
    PyArrayObject*)>;
using AddOverloads = std::map<AddOverloadsKey, AddOverload>;


#define ADD_ADD_OVERLOADS(                                         \
    npy_type,                                                      \
    type)                                                          \
{ AddOverloadsKey(npy_type, NPY_UINT8  ), add_##type##_uint8   },  \
{ AddOverloadsKey(npy_type, NPY_INT8   ), add_##type##_int8    },  \
{ AddOverloadsKey(npy_type, NPY_UINT16 ), add_##type##_uint16  },  \
{ AddOverloadsKey(npy_type, NPY_INT16  ), add_##type##_int16   },  \
{ AddOverloadsKey(npy_type, NPY_UINT32 ), add_##type##_uint32  },  \
{ AddOverloadsKey(npy_type, NPY_INT32  ), add_##type##_int32   },  \
{ AddOverloadsKey(npy_type, NPY_UINT64 ), add_##type##_uint64  },  \
{ AddOverloadsKey(npy_type, NPY_INT64  ), add_##type##_int64   },  \
{ AddOverloadsKey(npy_type, NPY_FLOAT32), add_##type##_float32 },  \
{ AddOverloadsKey(npy_type, NPY_FLOAT64), add_##type##_float64 },


static AddOverloads add_overloads = {
    ADD_ADD_OVERLOADS(NPY_UINT8, uint8)
    ADD_ADD_OVERLOADS(NPY_INT8, int8)
    ADD_ADD_OVERLOADS(NPY_UINT16, uint16)
    ADD_ADD_OVERLOADS(NPY_INT16, int16)
    ADD_ADD_OVERLOADS(NPY_UINT32, uint32)
    ADD_ADD_OVERLOADS(NPY_INT32, int32)
    ADD_ADD_OVERLOADS(NPY_UINT64, uint64)
    ADD_ADD_OVERLOADS(NPY_INT64, int64)
    ADD_ADD_OVERLOADS(NPY_FLOAT32, float32)
    ADD_ADD_OVERLOADS(NPY_FLOAT64, float64)
};


#undef ADD_ADD_OVERLOADS

} // namespace array_array
} // namespace detail


PyArrayObject* add(
    PyArrayObject* array,
    int64_t value)
{
    using namespace detail::array_int64;

    int data_type = PyArray_TYPE(array);
    AddOverloadsKey key(data_type);

    PyArrayObject* result{nullptr};

    if(add_overloads.find(key) == add_overloads.end()) {
        raise_unsupported_overload_exception(python_object(array));
        result = nullptr;
    }
    else {
        result = add_overloads[key](array, value);
    }

    return result;
}


PyArrayObject* add(
    int64_t value,
    PyArrayObject* array)
{
    return add(array, value);
}


PyArrayObject* add(
    PyArrayObject* array,
    float64_t value)
{
    using namespace detail::array_float64;

    int data_type = PyArray_TYPE(array);
    AddOverloadsKey key(data_type);

    PyArrayObject* result{nullptr};

    if(add_overloads.find(key) == add_overloads.end()) {
        raise_unsupported_overload_exception(python_object(array));
        result = nullptr;
    }
    else {
        result = add_overloads[key](array, value);
    }

    return result;
}


PyArrayObject* add(
    float64_t value,
    PyArrayObject* array)
{
    return add(array, value);
}


PyArrayObject* add(
    PyArrayObject* array1,
    PyArrayObject* array2)
{
    using namespace detail::array_array;

    int data_type1 = PyArray_TYPE(array1);
    int data_type2 = PyArray_TYPE(array1);
    AddOverloadsKey key(data_type1, data_type2);

    PyArrayObject* result{nullptr};

    if(add_overloads.find(key) == add_overloads.end()) {
        raise_unsupported_overload_exception(python_object(array1),
            python_object(array2));
        result = nullptr;
    }
    else {
        result = add_overloads[key](array1, array2);
    }

    return result;
}

} // namespace numpy
} // namespace python
} // namespace fern
