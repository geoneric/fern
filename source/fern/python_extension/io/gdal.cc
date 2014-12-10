#include "fern/python_extension/io/gdal.h"
#include <memory>
#include <gdal_priv.h>
#include "fern/io/gdal/gdal_data_type_traits.h"
#include "fern/io/gdal/gdal_type_traits.h"
#include "fern/python_extension/feature/masked_raster.h"


namespace bp = boost::python;
namespace fp = fern::python;


// TODO Handle special cases.
#define SWITCH_ON_GDAL_VALUE_TYPE( \
    value_type,                    \
    case_)                         \
switch(value_type) {               \
    case_(GDT_Byte, uint8_t)       \
    case_(GDT_UInt16, uint16_t)    \
    case_(GDT_Int16, int16_t)      \
    case_(GDT_UInt32, uint32_t)    \
    case_(GDT_Int32, int32_t)      \
    case_(GDT_Float32, float)      \
    case_(GDT_Float64, double)     \
    case GDT_Unknown: {            \
        assert(false);             \
        break;                     \
    }                              \
    case GDT_CInt16:               \
    case GDT_CInt32:               \
    case GDT_CFloat32:             \
    case GDT_CFloat64:             \
    case GDT_TypeCount: {          \
        assert(false);             \
        break;                     \
    }                              \
}


namespace fern {
namespace python {
namespace {

template<
    typename T>
void read_raster(
    GDALRasterBand& band,
    fern::MaskedRaster<T, 2>& raster)
{
    int const nr_rows = band.GetYSize();
    int const nr_cols = band.GetXSize();

    if(band.RasterIO(GF_Read, 0, 0, nr_cols, nr_rows, raster.data(),
            nr_cols, nr_rows, GDALTypeTraits<T>::data_type, 0, 0) != CE_None) {
        // TODO
        assert(false);
    }

    auto mask_flags(band.GetMaskFlags());

    if(!(mask_flags & GMF_ALL_VALID)) {
        assert(!(mask_flags & GMF_ALPHA));
        GDALRasterBand* mask_band = band.GetMaskBand();
        assert(mask_band->GetRasterDataType() == GDT_Byte);
        // The mask band has gdal data type GDT_Byte. A value of zero
        // means that the value must be masked.
        Array<GDALDataTypeTraits<GDT_Byte>::type, 2> mask(
            extents[nr_rows][nr_cols]);

        if(mask_band->RasterIO(GF_Read, 0, 0, nr_cols, nr_rows, mask.data(),
                nr_cols, nr_rows, GDT_Byte, 0, 0) != CE_None) {
            assert(false);
        }

        raster.set_mask(mask);
    }
}


#define CASE(                                                           \
        gdal_value_type,                                                \
        value_type)                                                     \
case gdal_value_type: {                                                 \
    using Raster = fern::MaskedRaster<value_type, 2>;                   \
    using Transformation = Raster::Transformation;                      \
    auto result_ptr(std::make_shared<Raster>(                           \
        extents[nr_rows][nr_cols], Transformation{transformation[0],    \
            transformation[1], transformation[3], transformation[5]})); \
    read_raster<value_type>(*band, *result_ptr);                        \
    result = std::make_shared<MaskedRaster>(result_ptr);                \
    break;                                                              \
}

std::shared_ptr<MaskedRaster> read_raster(
    std::string const& pathname)
{
    // Open raster dataset and obtain some properties.
    auto dataset = static_cast<GDALDataset *>(GDALOpen(pathname.c_str(),
        GA_ReadOnly));
    if(dataset == nullptr) {
        // TODO
        assert(dataset);
    }

    int nr_bands = dataset->GetRasterCount();
    if(nr_bands == 0) {
        // TODO
        assert(false);
    }

    double transformation[6];
    /* auto status = */ dataset->GetGeoTransform(transformation);
    // if(status != CE_None) {
    //     // TODO
    //     assert(false);
    // }

    int nr_rows = dataset->GetRasterYSize();
    int nr_cols = dataset->GetRasterXSize();


    // Read the first raster band.
    auto band = dataset->GetRasterBand(1);

    MaskedRasterHandle result;

    SWITCH_ON_GDAL_VALUE_TYPE(band->GetRasterDataType(), CASE)

    return result;
}

#undef CASE

} // anonymous namespace


bp::object read_raster(
    bp::str pathname_object)
{
    std::string pathname{bp::extract<std::string>(pathname_object)};

    return bp::object(read_raster(pathname));
}

} // namespace python
} // namespace fern
