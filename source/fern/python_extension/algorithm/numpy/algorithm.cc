#include "fern/python_extension/algorithm/numpy/algorithm.h"
#include "fern/feature/core/array_reference_traits.h"
#include "fern/algorithm.h"


namespace fern {

static void init_numpy()
{
    import_array();
}


PyArrayObject* add(
    PyArrayObject const* array_object1,
    PyArrayObject const* array_object2)
{
    init_numpy();

    // TODO Switch on number of dimensions.
    assert(PyArray_NDIM(array_object1) == 2);
    assert(PyArray_NDIM(array_object2) == 2);

    // TODO Switch on float size.
    assert(PyArray_ISFLOAT(array_object1));
    assert(PyArray_ITEMSIZE(array_object1) == 4);

    assert(PyArray_ISFLOAT(array_object2));
    assert(PyArray_ITEMSIZE(array_object2) == 4);

    // TODO Error handling.
    assert(PyArray_DIM(array_object1, 0) == PyArray_DIM(array_object2, 0));
    assert(PyArray_DIM(array_object1, 1) == PyArray_DIM(array_object2, 1));

    size_t const size1{static_cast<size_t>(PyArray_DIM(array_object1, 0))};
    size_t const size2{static_cast<size_t>(PyArray_DIM(array_object1, 1))};

    ArrayReference<float, 2> array_2d_reference1(
        static_cast<float*>(PyArray_DATA(const_cast<PyArrayObject*>(
            array_object1))), extents[size1][size2]);

    ArrayReference<float, 2> array_2d_reference2(
        static_cast<float*>(PyArray_DATA(const_cast<PyArrayObject*>(
            array_object2))), extents[size1][size2]);

    using result_value_type = algorithm::add::result_value_type<float, float>;

    PyArrayObject* result_object{(PyArrayObject*)(
        PyArray_SimpleNew(
            PyArray_NDIM(array_object1),
            PyArray_DIMS(const_cast<PyArrayObject*>(array_object1)),
            // TODO TypeTraits<T>::numpy_type_id
            NPY_FLOAT32))};

    ArrayReference<result_value_type, 2> result_reference(
        static_cast<result_value_type*>(PyArray_DATA(result_object)),
        extents[size1][size2]);

    algorithm::algebra::add(algorithm::parallel, array_2d_reference1,
        array_2d_reference2, result_reference);

    return result_object;
}


PyArrayObject* add(
    PyArrayObject const* array_object,
    PyFloatObject const* float_object)
{
    init_numpy();

    // TODO Switch on number of dimensions.
    assert(PyArray_NDIM(array_object) == 2);

    // TODO Switch on float size.
    assert(PyArray_ISFLOAT(array_object));
    assert(PyArray_ITEMSIZE(array_object) == 4);

    size_t const size1{static_cast<size_t>(PyArray_DIM(array_object, 0))};
    size_t const size2{static_cast<size_t>(PyArray_DIM(array_object, 1))};

    ArrayReference<float, 2> array_2d_reference(
        static_cast<float*>(PyArray_DATA(const_cast<PyArrayObject*>(
            array_object))), extents[size1][size2]);

    double const value(PyFloat_AS_DOUBLE(const_cast<PyFloatObject*>(
        float_object)));

    using result_value_type = algorithm::add::result_value_type<float, double>;

    PyArrayObject* result_object{(PyArrayObject*)(
        PyArray_SimpleNew(
            PyArray_NDIM(array_object),
            PyArray_DIMS(const_cast<PyArrayObject*>(array_object)),
            // TODO TypeTraits<T>::numpy_type_id
            NPY_FLOAT64))};

    ArrayReference<result_value_type, 2> result_reference(
        static_cast<result_value_type*>(PyArray_DATA(result_object)),
        extents[size1][size2]);

    algorithm::algebra::add(algorithm::parallel, array_2d_reference,
        value, result_reference);

    return result_object;
}

} // namespace fern
