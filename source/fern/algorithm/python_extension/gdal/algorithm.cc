#include "fern/algorithm/python_extension/gdal/algorithm.h"
// #include "fern/feature/core/array_reference_traits.h"
#include "fern/algorithm.h"


namespace fern {

// static void init_numpy()
// {
//     import_array();
// }


PyArrayObject* add(
    GDALRasterBand const* raster_band1,
    GDALRasterBand const* raster_band2)
{
    // http://trac.osgeo.org/gdal/wiki/rfc15_nodatabitmask
    // Mask can be a seperate band, or a special value.

    // if(masked1) {
    //     if(masked2) {
    //         // Call add with masked inputs. Write to masked output.
    //     }
    //     else {
    //         // Call add with masked input1 and non-masked input2.
    //         // Write to masked output.
    //     }
    // }
    // else if(masked2) {
    //     // Call add with masked input2 and non-masked input1.
    //     // Write to masked output.
    //     add(raster_band2, raster_band1);
    // }
    // else {
    //     // Call add with non-masked inputs. Write to non-masked output.
    // }

    // Check whether the dimensions of both bands are the same.
    //
    // No: Bail out
    //
    // For each band
    //     Check whether there are masked values
    //     No:
    //         Use a Raster
    //         Don't use an input no-data policy
    //         Don't use out of range policy
    //         Read band into raster
    //     Yes:
    //         If there is a seperate mask
    //             - Use a MaskedRaster
    //             - Use an appropriate input no-data policy testing the mask.
    //         If not
    //             - Use a Raster
    //             - Use an appropriate input no-data policy testing special
    //               value.
    //         Use an output no-data policy
    //         Use out of range policy
    //         Read band into raster
    //
    // Create result array, masked or not masked
    // If needed, fill result mask, configure input no-data policy.
    // Call algorithm

    return nullptr;
}


PyArrayObject* add(
    GDALRasterBand const* raster_band,
    double float_)
{
    return nullptr;
}


PyArrayObject* add(
    double float_,
    GDALRasterBand const* raster_band)
{
    return add(raster_band, float_);
}


double add(
    double float1,
    double float2)
{
    double result;
    algorithm::algebra::add(algorithm::parallel, float1, float2, result);
    return result;
}



    // Given a PyObject from something like the Python gdal.Open() wrapper 
    // the object looks like this in C: 
    // 
    // typedef struct { 
    //    PyObject_HEAD 
    //    void *ptr; 
    //    swig_type_info *ty; 
    //    int own; 
    //    PyObject *next; 
    // } PySwigObject; 
    // 
    // The "void *ptr" is the wrap C/C++ pointer and can be cast to 
    // GDALDatasetH or GDALDataset * as desired.  This approach is 
    // somewhat fragile it might be worth inserting externall C callable 
    // functions into the Python bindings that will safely return the 
    // corresponding C/C++ pointer from a Python object.  Basically this 
    // would be a variation on the SWIG_Python_ConvertPtrAndOwn() function 
    // produced in the existing SWIG bindings, but designed for external 
    // use. 

    // if(GDALDataset* gdal_dataset = extract_gdal_dataset(value1_object)) {
    //     GDALDriver* driver = gdal_dataset->GetDriver();
    //     char** metadata = driver->GetMetadata();
    //     assert(CSLFetchBoolean(metadata, GDAL_DCAP_CREATE, FALSE));

    //     // TODO Function generating name for temporary dataset.
    //     // TODO Add name of file to list dataset names to delete.
    //     std::string const filename{"blah.img"};
    //     int const nr_rows{gdal_dataset->GetRasterYSize()};
    //     int const nr_cols{gdal_dataset->GetRasterXSize()};
    //     int const nr_bands{1};
    //     GDALDataType data_type{GDT_Float64};
    //     char** options = nullptr;

    //     GDALDataset* raster_dataset = driver->Create(filename.c_str(), nr_cols,
    //         nr_rows, nr_bands, data_type, options);



    //     // TODO Copy meta data.
    //     //      - spatial reference
    //     //      - attributes
    //     //      - ...


    //     GDALRasterBand* raster_band = raster_dataset->GetRasterBand(1);



    //     // TODO Fill band with values.





    //     GDALClose(raster_dataset);

    //     std::cout << filename << std::endl;

    //     // Call gdal Python extension to open the file and return the
    //     // instance.
    //     // raster = gdal.Open("raster-1.img", GA_ReadOnly)
    //     PyObject* main_module = PyImport_AddModule("__main__");  // BR
    //     PyObject* globals = PyModule_GetDict(main_module);  // BR
    //     PyObject* locals = PyDict_New();  // NR

    //     std::string command =
    //         "gdal.Open(\"" + filename + "\", GA_ReadOnly)\n";
    //     PyCodeObject* compiled_command = reinterpret_cast<PyCodeObject*>(
    //         Py_CompileString(command.c_str(), "<string>", Py_eval_input));
    //     assert(compiled_command != nullptr);

    //     result = PyEval_EvalCode(compiled_command, globals, locals);

    //     Py_DECREF(locals);

    //     assert(result != nullptr);
    //     assert(result != Py_None);
    //     assert(is_gdal_dataset_object(result));
    // }
    // else {
    //     std::cout << "nope" << std::endl;
    // }

    // PySwigObject* swig_object = static_cast<swig_object*>(value1_object);
    // GDALDataset* raster_dataset = static_cast<GDALDataset*>(value1_object->ptr);


/// SwigPyObject* add(
///     SwigPyObject const* array_object1,
///     SwigPyObject const* array_object2)
/// {
///     /// init_numpy();
/// 
///     /// // TODO Switch on number of dimensions.
///     /// assert(PyArray_NDIM(array_object1) == 2);
///     /// assert(PyArray_NDIM(array_object2) == 2);
/// 
///     /// // TODO Switch on float size.
///     /// assert(PyArray_ISFLOAT(array_object1));
///     /// assert(PyArray_ITEMSIZE(array_object1) == 4);
/// 
///     /// assert(PyArray_ISFLOAT(array_object2));
///     /// assert(PyArray_ITEMSIZE(array_object2) == 4);
/// 
///     /// // TODO Error handling.
///     /// assert(PyArray_DIM(array_object1, 0) == PyArray_DIM(array_object2, 0));
///     /// assert(PyArray_DIM(array_object1, 1) == PyArray_DIM(array_object2, 1));
/// 
///     /// size_t const size1{static_cast<size_t>(PyArray_DIM(array_object1, 0))};
///     /// size_t const size2{static_cast<size_t>(PyArray_DIM(array_object1, 1))};
/// 
///     /// ArrayReference<float, 2> array_2d_reference1(
///     ///     static_cast<float*>(PyArray_DATA(const_cast<PyArrayObject*>(
///     ///         array_object1))), extents[size1][size2]);
/// 
///     /// ArrayReference<float, 2> array_2d_reference2(
///     ///     static_cast<float*>(PyArray_DATA(const_cast<PyArrayObject*>(
///     ///         array_object2))), extents[size1][size2]);
/// 
///     /// init_numpy();
/// 
///     /// using result_value_type = algorithm::add::result_value_type<float, float>;
/// 
///     /// PyArrayObject* result_object{(PyArrayObject*)(
///     ///     PyArray_SimpleNew(
///     ///         PyArray_NDIM(array_object1),
///     ///         PyArray_DIMS(const_cast<PyArrayObject*>(array_object1)),
///     ///         // TODO TypeTraits<T>::numpy_type_id
///     ///         NPY_FLOAT32))};
/// 
///     /// ArrayReference<result_value_type, 2> result_reference(
///     ///     static_cast<result_value_type*>(PyArray_DATA(result_object)),
///     ///     extents[size1][size2]);
/// 
///     /// algorithm::algebra::add(algorithm::parallel, array_2d_reference1,
///     ///     array_2d_reference2, result_reference);
/// 
///     SwigPyObject* result_object = nullptr;
///     return result_object;
/// }
/// 
/// 
/// SwigPyObject* add(
///     SwigPyObject const* array_object,
///     PyFloatObject const* float_object)
/// {
///     /// init_numpy();
/// 
///     /// // TODO Switch on number of dimensions.
///     /// assert(PyArray_NDIM(array_object) == 2);
/// 
///     /// // TODO Switch on float size.
///     /// assert(PyArray_ISFLOAT(array_object));
///     /// assert(PyArray_ITEMSIZE(array_object) == 4);
/// 
///     /// size_t const size1{static_cast<size_t>(PyArray_DIM(array_object, 0))};
///     /// size_t const size2{static_cast<size_t>(PyArray_DIM(array_object, 1))};
/// 
///     /// ArrayReference<float, 2> array_2d_reference(
///     ///     static_cast<float*>(PyArray_DATA(const_cast<PyArrayObject*>(
///     ///         array_object))), extents[size1][size2]);
/// 
///     /// double const value(PyFloat_AS_DOUBLE(const_cast<PyFloatObject*>(
///     ///     float_object)));
/// 
///     /// using result_value_type = algorithm::add::result_value_type<float, double>;
/// 
///     /// PyArrayObject* result_object{(PyArrayObject*)(
///     ///     PyArray_SimpleNew(
///     ///         PyArray_NDIM(array_object),
///     ///         PyArray_DIMS(const_cast<PyArrayObject*>(array_object)),
///     ///         // TODO TypeTraits<T>::numpy_type_id
///     ///         NPY_FLOAT64))};
/// 
///     /// ArrayReference<result_value_type, 2> result_reference(
///     ///     static_cast<result_value_type*>(PyArray_DATA(result_object)),
///     ///     extents[size1][size2]);
/// 
///     /// algorithm::algebra::add(algorithm::parallel, array_2d_reference,
///     ///     value, result_reference);
/// 
///     SwigPyObject* result_object = nullptr;
///     return result_object;
/// }

} // namespace fern
