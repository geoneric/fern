#include "fern/python_extension/algorithm/gdal/add.h"
#include <functional>
#include <map>
#include <tuple>
#include <gdal_priv.h>
#include "fern/feature/core/array_reference_traits.h"
#include "fern/feature/core/array_traits.h"
#include "fern/algorithm/algebra/elementary/add.h"
#include "fern/io/gdal/gdal_type_traits.h"
#include "fern/python_extension/core/error.h"
#include "fern/python_extension/algorithm/core/macro.h"
#include "fern/python_extension/algorithm/numpy/numpy_type_traits.h"
#include "fern/python_extension/algorithm/gdal/util.h"


namespace fern {
namespace python {
namespace detail {

static void init_numpy()
{
    import_array();
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


#define BFO BINARY_FUNCTION_OVERLOAD

#define ADD_OVERLOADS(                                           \
    algorithm,                                                   \
    type)                                                        \
BFO(algorithm, GDALRasterBand*, type, GDALRasterBand*, uint8)    \
BFO(algorithm, GDALRasterBand*, type, GDALRasterBand*, uint16)   \
BFO(algorithm, GDALRasterBand*, type, GDALRasterBand*, int16)    \
BFO(algorithm, GDALRasterBand*, type, GDALRasterBand*, uint32)   \
BFO(algorithm, GDALRasterBand*, type, GDALRasterBand*, int32)    \
BFO(algorithm, GDALRasterBand*, type, GDALRasterBand*, float32)  \
BFO(algorithm, GDALRasterBand*, type, GDALRasterBand*, float64)

ADD_OVERLOADS(add, uint8)
ADD_OVERLOADS(add, uint16)
ADD_OVERLOADS(add, int16)
ADD_OVERLOADS(add, uint32)
ADD_OVERLOADS(add, int32)
ADD_OVERLOADS(add, float32)
ADD_OVERLOADS(add, float64)

#undef ADD_OVERLOADS
#undef BFO


using AddOverloadsKey = std::tuple<GDALDataType, GDALDataType>;
using AddOverload = std::function<PyArrayObject*(GDALRasterBand*,
    GDALRasterBand*)>;
using AddOverloads = std::map<AddOverloadsKey, AddOverload>;


#define ADD_ADD_OVERLOADS(                                          \
    gdal_type,                                                      \
    type)                                                           \
{ AddOverloadsKey(gdal_type, GDT_Byte   ), add_##type##_uint8   },  \
{ AddOverloadsKey(gdal_type, GDT_UInt16 ), add_##type##_uint16  },  \
{ AddOverloadsKey(gdal_type, GDT_Int16  ), add_##type##_int16   },  \
{ AddOverloadsKey(gdal_type, GDT_UInt32 ), add_##type##_uint32  },  \
{ AddOverloadsKey(gdal_type, GDT_Int32  ), add_##type##_int32   },  \
{ AddOverloadsKey(gdal_type, GDT_Float32), add_##type##_float32 },  \
{ AddOverloadsKey(gdal_type, GDT_Float64), add_##type##_float64 },

static AddOverloads add_overloads = {
    ADD_ADD_OVERLOADS(GDT_Byte, uint8)
    ADD_ADD_OVERLOADS(GDT_UInt16, uint16)
    ADD_ADD_OVERLOADS(GDT_Int16, int16)
    ADD_ADD_OVERLOADS(GDT_UInt32, uint32)
    ADD_ADD_OVERLOADS(GDT_Int32, int32)
    ADD_ADD_OVERLOADS(GDT_Float32, float32)
    ADD_ADD_OVERLOADS(GDT_Float64, float64)
};

#undef ADD_ADD_OVERLOADS

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


#define BFO BINARY_FUNCTION_OVERLOAD

#define ADD_OVERLOADS(                                            \
    algorithm,                                                    \
    type)                                                         \
BFO(algorithm, GDALRasterBand*, type, uint8_t const&, uint8)      \
BFO(algorithm, GDALRasterBand*, type, uint16_t const&, uint16)    \
BFO(algorithm, GDALRasterBand*, type, int16_t const&, int16)      \
BFO(algorithm, GDALRasterBand*, type, uint32_t const&, uint32)    \
BFO(algorithm, GDALRasterBand*, type, int32_t const&, int32)      \
BFO(algorithm, GDALRasterBand*, type, float32_t const&, float32)  \
BFO(algorithm, GDALRasterBand*, type, float64_t const&, float64)

ADD_OVERLOADS(add, uint8)
ADD_OVERLOADS(add, uint16)
ADD_OVERLOADS(add, int16)
ADD_OVERLOADS(add, uint32)
ADD_OVERLOADS(add, int32)
ADD_OVERLOADS(add, float32)
ADD_OVERLOADS(add, float64)

#undef ADD_OVERLOADS
#undef BFO

} // namespace raster_band_number


namespace raster_band_float64 {

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
    { AddOverloadsKey(GDT_Float64 ), raster_band_number::add_float64_float64 }
};

} // namespace raster_band_float64
} // namespace detail


PyArrayObject* add(
    GDALRasterBand* raster_band,
    float64_t value)
{
    using namespace detail::raster_band_float64;

    GDALDataType data_type = raster_band->GetRasterDataType();
    AddOverloadsKey key(data_type);

    PyArrayObject* result{nullptr};

    if(add_overloads.find(key) == add_overloads.end()) {
        raise_unsupported_overload_exception(to_string(data_type));
        result = nullptr;
    }
    else {
        result = add_overloads[key](raster_band, value);
    }

    return result;
}


PyArrayObject* add(
    float64_t value,
    GDALRasterBand* raster_band)
{
    return add(raster_band, value);
}


PyArrayObject* add(
    GDALRasterBand* raster_band,
    PyArrayObject* array)
{
    // TODO
    return nullptr;
}


PyArrayObject* add(
    PyArrayObject* array,
    GDALRasterBand* raster_band)
{
    return add(raster_band, array);
}


PyArrayObject* add(
    GDALRasterBand* raster_band1,
    GDALRasterBand* raster_band2)
{
    using namespace detail::raster_band_raster_band;

    GDALDataType data_type1 = raster_band1->GetRasterDataType();
    GDALDataType data_type2 = raster_band2->GetRasterDataType();
    AddOverloadsKey key(data_type1, data_type2);

    PyArrayObject* result{nullptr};

    if(add_overloads.find(key) == add_overloads.end()) {
        raise_unsupported_overload_exception(to_string(data_type1),
            to_string(data_type2));
        result = nullptr;
    }
    else {
        result = add_overloads[key](raster_band1, raster_band2);
    }

    return result;
}

} // namespace python
} // namespace fern
