#include "fern/algorithm/python_extension/gdal/algorithm.h"
#include <functional>
#include <map>
#include <tuple>
#include "gdal_priv.h"
#include "fern/io/gdal/gdal_type_traits.h"
#include "fern/feature/core/array_reference_traits.h"
#include "fern/feature/core/array_traits.h"
#include "fern/feature/core/raster_traits.h"
#include "fern/algorithm/python_extension/gdal/numpy_type_traits.h"
#include "fern/algorithm/python_extension/gdal/error.h"
#include "fern/algorithm.h"


namespace fern {
namespace detail {

static void init_numpy()
{
    import_array();
}


std::map<GDALDataType, std::string> gdal_type_names {
    { GDT_Byte, "GDT_Byte" },
    { GDT_UInt16, "GDT_UInt16" },
    { GDT_Int16, "GDT_Int16" },
    { GDT_UInt32, "GDT_UInt32" },
    { GDT_Int32, "GDT_Int32" },
    { GDT_Float32, "GDT_Float32" },
    { GDT_Float64, "GDT_Float64" },
    { GDT_CInt16, "GDT_CInt16" },
    { GDT_CInt32, "GDT_CInt32" },
    { GDT_CFloat32, "GDT_CFloat32" },
    { GDT_CFloat64, "GDT_CFloat64" }
};


std::string to_string(
    GDALDataType const& data_type)
{
    assert(gdal_type_names.find(data_type) != gdal_type_names.end());
    return gdal_type_names[data_type];
}


namespace raster_band_raster_band {

template<
    typename Value1,
    typename Value2>
PyArrayObject* add(
    GDALRasterBand* raster_band1,
    GDALRasterBand* raster_band2)
{
    // TODO
    assert(raster_band1->GetXSize() == raster_band2->GetXSize());
    assert(raster_band1->GetYSize() == raster_band2->GetYSize());
    // TODO: check band1 and band2 have the same transform.

    init_numpy();

    // Read raster bands into rasters.
    int const nr_rows = raster_band1->GetYSize();
    int const nr_cols = raster_band1->GetXSize();
    auto extents = fern::extents[nr_rows][nr_cols];

    /// double geo_transform[6];
    /// raster_band1->GetDataset()->GetGeoTransform(geo_transform);
    /// double const cell_width = geo_transform[1];
    /// double const cell_height = std::abs(geo_transform[5]);
    /// double const west = geo_transform[0];
    /// double const north = geo_transform[3];

    // typename Raster<Value1, 2>::Transformation transformation{{west, cell_width, north,
    //     cell_height}};
    Array<Value1, 2> array1(extents); // , transformation);
    Array<Value2, 2> array2(extents); // , transformation);

    if(raster_band1->RasterIO(GF_Read, 0, 0, nr_cols, nr_rows, array1.data(),
            nr_cols, nr_rows, GDALTypeTraits<Value1>::data_type, 0, 0) !=
                CE_None) {
        // TODO
        assert(false);
    }

    if(raster_band2->RasterIO(GF_Read, 0, 0, nr_cols, nr_rows, array2.data(),
            nr_cols, nr_rows, GDALTypeTraits<Value2>::data_type, 0, 0) !=
                CE_None) {
        // TODO
        assert(false);
    }

    using result_value_type = algorithm::add::result_value_type<Value1, Value2>;

    npy_intp dimensions[] = {nr_rows, nr_cols};
    PyArrayObject* result_object{(PyArrayObject*)(
        PyArray_SimpleNew(
            2,  // NDIM
            dimensions,
            NumpyTypeTraits<result_value_type>::data_type))};

    ArrayReference<result_value_type, 2> result_reference(
        static_cast<result_value_type*>(PyArray_DATA(result_object)), extents);

    algorithm::algebra::add(algorithm::parallel, array1, array2,
        result_reference);

    return result_object;




    // If the input contains no-data, handle it.
    // - Special no-data values.
    // - Special no-data mask raster band.

    // If at least one of the inputs contains no-data:
    // - Result must be able to contain no-data: masked array.
    // - No-data values of all relevant inputs must be put in the output
    //   mask.
    // - An input-no-data policy must be set-up to read the mask.
    // - An output-no-data policy must be set-up to write the mask.
    //
    // If none of the inputs contains no-data:
    // - Result must be able to contain no-data: masked array.
    // - We can use SkipNoData as the input-no-data policy.
    // - An output-no-data policy must be set-up to write the mask.
    //
    // This assumes we always want to be notified of no-data. Make this a
    // configurable setting?


    // Call fern::algorithm::add.


    return nullptr;


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
}


PyArrayObject* add_unsupported_unsupported(
    GDALRasterBand* raster_band1,
    GDALRasterBand* /* raster_band2 */)
{
    GDALDataType data_type = raster_band1->GetRasterDataType();

    raise_unsupported_argument_type_exception(to_string(data_type));

    return nullptr;
}


PyArrayObject* unsupported1(
    GDALRasterBand* raster_band1,
    GDALRasterBand* /* raster_band2 */)
{
    GDALDataType data_type = raster_band1->GetRasterDataType();
    raise_unsupported_argument_type_exception(to_string(data_type));
    return nullptr;
}


PyArrayObject* unsupported2(
    GDALRasterBand* /* raster_band1 */,
    GDALRasterBand* raster_band2)
{
    GDALDataType data_type = raster_band2->GetRasterDataType();
    raise_unsupported_argument_type_exception(to_string(data_type));
    return nullptr;
}


#define ADD_OVERLOAD(                                                    \
    algorithm,                                                           \
    type1,                                                               \
    type2)                                                               \
PyArrayObject* algorithm##_##type1##_##type2(                            \
    GDALRasterBand* raster_band1,                                        \
    GDALRasterBand* raster_band2)                                        \
{                                                                        \
    return algorithm<type1##_t, type2##_t>(raster_band1, raster_band2);  \
}


#define ADD_OVERLOADS(                  \
    algorithm,                          \
    type)                               \
ADD_OVERLOAD(algorithm, type, uint8)    \
ADD_OVERLOAD(algorithm, type, uint16)   \
ADD_OVERLOAD(algorithm, type, int16)    \
ADD_OVERLOAD(algorithm, type, uint32)   \
ADD_OVERLOAD(algorithm, type, int32)    \
ADD_OVERLOAD(algorithm, type, float32)  \
ADD_OVERLOAD(algorithm, type, float64)


ADD_OVERLOADS(add, uint8)
ADD_OVERLOADS(add, uint16)
ADD_OVERLOADS(add, int16)
ADD_OVERLOADS(add, uint32)
ADD_OVERLOADS(add, int32)
ADD_OVERLOADS(add, float32)
ADD_OVERLOADS(add, float64)


#undef ADD_OVERLOADS
#undef ADD_OVERLOAD


using AddOverloadsKey = std::tuple<GDALDataType, GDALDataType>;
using AddOverload = std::function<PyArrayObject*(GDALRasterBand*,
    GDALRasterBand*)>;
using AddOverloads = std::map<AddOverloadsKey, AddOverload>;


#define ADD_SUPPORTED_ADD_OVERLOADS(                                \
    gdal_type,                                                      \
    type)                                                           \
{ AddOverloadsKey(gdal_type, GDT_Byte   ), add_##type##_uint8   },  \
{ AddOverloadsKey(gdal_type, GDT_UInt16 ), add_##type##_uint16  },  \
{ AddOverloadsKey(gdal_type, GDT_Int16  ), add_##type##_int16   },  \
{ AddOverloadsKey(gdal_type, GDT_UInt32 ), add_##type##_uint32  },  \
{ AddOverloadsKey(gdal_type, GDT_Int32  ), add_##type##_int32   },  \
{ AddOverloadsKey(gdal_type, GDT_Float32), add_##type##_float32 },  \
{ AddOverloadsKey(gdal_type, GDT_Float64), add_##type##_float64 },


#define ADD_UNSUPPORTED_ADD_OVERLOADS(                       \
    gdal_type)                                               \
{ AddOverloadsKey(GDT_CInt16  , gdal_type), unsupported1 },  \
{ AddOverloadsKey(GDT_CInt32  , gdal_type), unsupported1 },  \
{ AddOverloadsKey(GDT_CFloat32, gdal_type), unsupported1 },  \
{ AddOverloadsKey(GDT_CFloat64, gdal_type), unsupported1 },  \
{ AddOverloadsKey(gdal_type, GDT_CInt16  ), unsupported2 },  \
{ AddOverloadsKey(gdal_type, GDT_CInt32  ), unsupported2 },  \
{ AddOverloadsKey(gdal_type, GDT_CFloat32), unsupported2 },  \
{ AddOverloadsKey(gdal_type, GDT_CFloat64), unsupported2 },


#define ADD_ADD_OVERLOADS(                    \
    gdal_type,                                \
    type)                                     \
ADD_SUPPORTED_ADD_OVERLOADS(gdal_type, type)  \
ADD_UNSUPPORTED_ADD_OVERLOADS(gdal_type)


static AddOverloads add_overloads = {
    ADD_ADD_OVERLOADS(GDT_Byte, uint8)
    ADD_ADD_OVERLOADS(GDT_UInt16, uint16)
    ADD_ADD_OVERLOADS(GDT_Int16, int16)
    ADD_ADD_OVERLOADS(GDT_UInt32, uint32)
    ADD_ADD_OVERLOADS(GDT_Int32, int32)
    ADD_ADD_OVERLOADS(GDT_Float32, float32)
    ADD_ADD_OVERLOADS(GDT_Float64, float64)
    {AddOverloadsKey(GDT_Unknown, GDT_Unknown), add_unsupported_unsupported }
};


#undef ADD_ADD_OVERLOADS
#undef ADD_UNSUPPORTED_ADD_OVERLOADS
#undef ADD_SUPPORTED_ADD_OVERLOADS

} // namespace raster_band_raster_band


namespace raster_band_number {

template<
    typename Value1,
    typename Value2>
PyArrayObject* add(
    GDALRasterBand* raster_band,
    Value2 const& value)
{
    init_numpy();

    // Read raster band into raster.
    int const nr_rows = raster_band->GetYSize();
    int const nr_cols = raster_band->GetXSize();
    auto extents = fern::extents[nr_rows][nr_cols];

    Array<Value1, 2> array(extents);

    if(raster_band->RasterIO(GF_Read, 0, 0, nr_cols, nr_rows, array.data(),
            nr_cols, nr_rows, GDALTypeTraits<Value1>::data_type, 0, 0) !=
                CE_None) {
        // TODO
        assert(false);
    }

    using result_value_type = algorithm::add::result_value_type<Value1, Value2>;

    npy_intp dimensions[] = {nr_rows, nr_cols};
    PyArrayObject* result_object{(PyArrayObject*)(
        PyArray_SimpleNew(
            2,  // NDIM
            dimensions,
            NumpyTypeTraits<result_value_type>::data_type))};

    ArrayReference<result_value_type, 2> result_reference(
        static_cast<result_value_type*>(PyArray_DATA(result_object)), extents);

    algorithm::algebra::add(algorithm::parallel, array, value,
        result_reference);

    return result_object;
}


#define ADD_OVERLOAD(                                            \
    algorithm,                                                   \
    type1,                                                       \
    type2)                                                       \
PyArrayObject* algorithm##_##type1##_##type2(                    \
    GDALRasterBand* raster_band,                                 \
    type2##_t const& value)                                      \
{                                                                \
    return algorithm<type1##_t, type2##_t>(raster_band, value);  \
}


#define ADD_OVERLOADS(                  \
    algorithm,                          \
    type)                               \
ADD_OVERLOAD(algorithm, type, uint8)    \
ADD_OVERLOAD(algorithm, type, uint16)   \
ADD_OVERLOAD(algorithm, type, int16)    \
ADD_OVERLOAD(algorithm, type, uint32)   \
ADD_OVERLOAD(algorithm, type, int32)    \
ADD_OVERLOAD(algorithm, type, float32)  \
ADD_OVERLOAD(algorithm, type, float64)


ADD_OVERLOADS(add, uint8)
ADD_OVERLOADS(add, uint16)
ADD_OVERLOADS(add, int16)
ADD_OVERLOADS(add, uint32)
ADD_OVERLOADS(add, int32)
ADD_OVERLOADS(add, float32)
ADD_OVERLOADS(add, float64)


#undef ADD_OVERLOADS
#undef ADD_OVERLOAD

} // namespace raster_band_number


namespace raster_band_float64 {

PyArrayObject* unsupported1(
    GDALRasterBand* raster_band,
    float64_t const& /* value */)
{
    GDALDataType data_type = raster_band->GetRasterDataType();
    raise_unsupported_argument_type_exception(to_string(data_type));
    return nullptr;
}


using AddOverloadsKey = GDALDataType;
using AddOverload = std::function<PyArrayObject*(GDALRasterBand*, float64_t)>;
using AddOverloads = std::map<AddOverloadsKey, AddOverload>;


static AddOverloads add_overloads = {
    { AddOverloadsKey(GDT_Byte    ), raster_band_number::add_uint8_float64   },
    { AddOverloadsKey(GDT_UInt16  ), raster_band_number::add_uint16_float64  },
    { AddOverloadsKey(GDT_Int16   ), raster_band_number::add_int16_float64   },
    { AddOverloadsKey(GDT_UInt32  ), raster_band_number::add_uint32_float64  },
    { AddOverloadsKey(GDT_Int32   ), raster_band_number::add_int32_float64   },
    { AddOverloadsKey(GDT_Float32 ), raster_band_number::add_float32_float64 },
    { AddOverloadsKey(GDT_Float64 ), raster_band_number::add_float64_float64 },
    { AddOverloadsKey(GDT_CInt16  ), unsupported1 },
    { AddOverloadsKey(GDT_CInt32  ), unsupported1 },
    { AddOverloadsKey(GDT_CFloat32), unsupported1 },
    { AddOverloadsKey(GDT_CFloat64), unsupported1 },
};

} // namespace raster_band_float64
} // namespace detail


PyArrayObject* add(
    GDALRasterBand* raster_band1,
    GDALRasterBand* raster_band2)
{
    using namespace detail::raster_band_raster_band;

    GDALDataType data_type1 = raster_band1->GetRasterDataType();
    GDALDataType data_type2 = raster_band2->GetRasterDataType();

    AddOverloadsKey key(data_type1, data_type2);
    return add_overloads[key](raster_band1, raster_band2);
}


PyArrayObject* add(
    GDALRasterBand* raster_band,
    float64_t value)
{
    using namespace detail::raster_band_float64;

    GDALDataType data_type = raster_band->GetRasterDataType();
    AddOverloadsKey key(data_type);
    return add_overloads[key](raster_band, value);
}


PyArrayObject* add(
    float64_t value,
    GDALRasterBand* raster_band)
{
    return add(raster_band, value);
}


double add(
    double value1,
    double value2)
{
    double result;
    algorithm::algebra::add(algorithm::parallel, value1, value2, result);
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
