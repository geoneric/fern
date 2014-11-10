#include "fern/python_extension/algorithm/gdal/slope.h"
#include <functional>
#include <map>
#include <gdal_priv.h>
#include "fern/core/types.h"
#include "fern/feature/core/array_reference_traits.h"
#include "fern/feature/core/raster_traits.h"
#include "fern/algorithm/space/focal/slope.h"
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


namespace raster_band {

template<
    typename Value>
PyArrayObject* slope(
    GDALRasterBand* raster_band)
{
    init_numpy();

    // Read raster band into raster.
    int const nr_rows = raster_band->GetYSize();
    int const nr_cols = raster_band->GetXSize();
    auto extents = fern::extents[nr_rows][nr_cols];

    double geo_transform[6];
    raster_band->GetDataset()->GetGeoTransform(geo_transform);
    double const cell_width = geo_transform[1];
    double const cell_height = std::abs(geo_transform[5]);
    double const west = geo_transform[0];
    double const north = geo_transform[3];

    typename Raster<Value, 2>::Transformation transformation{
        {west, cell_width, north, cell_height}};
    Raster<Value, 2> raster(extents, transformation);

    if(raster_band->RasterIO(GF_Read, 0, 0, nr_cols, nr_rows, raster.data(),
            nr_cols, nr_rows, GDALTypeTraits<Value>::data_type, 0, 0) !=
                CE_None) {
        // TODO
        assert(false);
    }

    using result_value_type = Value;

    npy_intp dimensions[] = {nr_rows, nr_cols};
    PyArrayObject* result_object{(PyArrayObject*)(
        PyArray_SimpleNew(
            2,  // NDIM
            dimensions,
            NumpyTypeTraits<result_value_type>::data_type))};

    ArrayReference<result_value_type, 2> result_reference(
        static_cast<result_value_type*>(PyArray_DATA(result_object)), extents);

    algorithm::space::slope(algorithm::parallel, raster, result_reference);

    return result_object;
}


#define UFO UNARY_FUNCTION_OVERLOAD

UFO(slope, GDALRasterBand*, float32)
UFO(slope, GDALRasterBand*, float64)

#undef UFO

} // namespace raster_band


namespace raster_band {

using AddOverloadsKey = GDALDataType;
using AddOverload = std::function<PyArrayObject*(GDALRasterBand*)>;
using SlopeOverloads = std::map<AddOverloadsKey, AddOverload>;


static SlopeOverloads slope_overloads = {
    { AddOverloadsKey(GDT_Float32), raster_band::slope_float32 },
    { AddOverloadsKey(GDT_Float64), raster_band::slope_float64 }
};

} // namespace raster_band
} // namespace detail


PyArrayObject* slope(
    GDALRasterBand* raster_band)
{
    using namespace detail::raster_band;

    GDALDataType data_type = raster_band->GetRasterDataType();
    AddOverloadsKey key(data_type);

    PyArrayObject* result{nullptr};

    if(slope_overloads.find(key) == slope_overloads.end()) {
        raise_unsupported_overload_exception(to_string(data_type));
        result = nullptr;
    }
    else {
        result = slope_overloads[key](raster_band);
    }

    return result;
}

} // namespace python
} // namespace fern
