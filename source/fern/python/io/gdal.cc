// -----------------------------------------------------------------------------
// Fern © Geoneric
//
// This file is part of Geoneric Fern which is available under the terms of
// the GNU General Public License (GPL), version 2. If you do not want to
// be bound by the terms of the GPL, you may purchase a proprietary license
// from Geoneric (http://www.geoneric.eu/contact).
// -----------------------------------------------------------------------------
#include "fern/python/io/gdal.h"
#include <memory>
#include <gdal_priv.h>
#include "fern/python/feature/detail/data_customization_point/masked_raster.h"
#include "fern/feature/core/array.h"
#include "fern/algorithm/policy/mark_no_data.h"
#include "fern/io/gdal/dataset.h"
#include "fern/io/gdal/gdal_value_type_traits.h"
#include "fern/io/gdal/read.h"
#include "fern/io/gdal/value_type_traits.h"


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


#define SWITCH_ON_GDAL_SUPPORTED_VALUE_TYPE( \
    value_type,                 \
    case_)                      \
switch(value_type) {            \
    case_(VT_UINT8, uint8_t)    \
    case_(VT_UINT16, uint16_t)  \
    case_(VT_INT16, int16_t)    \
    case_(VT_UINT32, uint32_t)  \
    case_(VT_INT32, int32_t)    \
    case_(VT_FLOAT32, float)    \
    case_(VT_FLOAT64, double)   \
    case VT_BOOL:               \
    case VT_CHAR:               \
    case VT_INT8:               \
    case VT_UINT64:             \
    case VT_INT64:              \
    case VT_STRING: {           \
        assert(false);          \
    }                           \
}


namespace fern {
namespace python {

#define CASE(                                                                \
        gdal_value_type,                                                     \
        value_type)                                                          \
case gdal_value_type: {                                                      \
    using OutputNoDataPolicy = fern::algorithm::MarkNoData<                  \
        detail::MaskedRaster<value_type>>;                                   \
    auto result_ptr(std::make_shared<detail::MaskedRaster<value_type>>(      \
        std::make_tuple(static_cast<size_t>(nr_rows),                        \
            static_cast<size_t>(nr_cols)),                                   \
        std::make_tuple(transformation[0], transformation[3]),               \
        std::make_tuple(transformation[1], transformation[5])));             \
    OutputNoDataPolicy output_no_data_policy(*result_ptr);                   \
    io::gdal::read(output_no_data_policy, DataName(pathname), *result_ptr);  \
    result = std::make_shared<MaskedRaster>(result_ptr);                     \
    break;                                                                   \
}


MaskedRasterHandle read_raster(
    std::string const& pathname)
{
    GDALDataType gdal_value_type;
    int nr_rows, nr_cols;
    double transformation[6];

    {
        auto dataset = io::gdal::open_dataset(pathname, GA_ReadOnly);
        auto band = raster_band(dataset, 1);
        gdal_value_type = band->GetRasterDataType();
        nr_rows = dataset->GetRasterYSize();
        nr_cols = dataset->GetRasterXSize();
        /* auto status = */ dataset->GetGeoTransform(transformation);
    }

    MaskedRasterHandle result;

    SWITCH_ON_GDAL_VALUE_TYPE(gdal_value_type, CASE)

    return result;
}

#undef CASE


template<
    typename T>
void write_raster(
    detail::MaskedRaster<T> const& raster,
    std::string const& name,
    GDALDriver* driver)
{
    int const nr_rows{static_cast<int>(std::get<0>(raster.sizes()))};
    int const nr_cols{static_cast<int>(std::get<1>(raster.sizes()))};
    int const nr_bands{1};
    GDALDataType value_type{io::gdal::ValueTypeTraits<T>::gdal_value_type};
    auto dataset = driver->Create(name.c_str(), nr_cols, nr_rows, nr_bands,
        value_type, nullptr);
    // TODO
    assert(dataset);

    // Set some metadata.
    double transformation[]{
        std::get<0>(raster.origin()),  // Top left x.
        std::get<0>(raster.cell_sizes()),  // Cell width.
        0.0,
        std::get<1>(raster.origin()),  // Top left y.
        0.0,
        std::get<1>(raster.cell_sizes())};  // Cell height.
    dataset->SetGeoTransform(transformation);

    // Write values to the raster band.
    auto band = dataset->GetRasterBand(1);

    band->SetNoDataValue(no_data_value<T>());

    if(band->RasterIO(GF_Write, 0, 0, nr_cols, nr_rows,
            const_cast<T*>(raster.data()),
            nr_cols, nr_rows, value_type, 0, 0) != CE_None) {
        // TODO
        assert(false);
    }

    // // Write mask band.
    // if(band->CreateMaskBand(GMF_PER_DATASET) != CE_None) {
    //     // TODO
    //     assert(false);
    // }

    // auto mask_band = band->GetMaskBand();
    // assert(mask_band->GetRasterDataType() == GDT_Byte);
    // // The mask band has gdal data type GDT_Byte. A value of zero
    // // means that the value must be masked.
    // Array<typename GDALDataTypeTraits<GDT_Byte>::type, 2> mask(
    //     extents[nr_rows][nr_cols], 1);
    // // raster.mask(mask);

    // auto elements = raster.data();
    // auto mask_elements = mask.data();

    // for(size_t i = 0; i < raster.size(); ++i) {
    //     if(is_no_data(elements[i])) {
    //         mask_elements[i] = 0;
    //     }
    // }

    // if(mask_band->RasterIO(GF_Write, 0, 0, nr_cols, nr_rows, mask.data(),
    //         nr_cols, nr_rows, GDT_Byte, 0, 0) != CE_None) {
    //     // TODO
    //     assert(false);
    // }

    // Close dataset.
    GDALClose(dataset);
}


#define CASE(                          \
        value_type_enum,               \
        value_type)                    \
case value_type_enum: {                \
    write_raster(                      \
        raster->raster<value_type>(),  \
        name, driver);                 \
    break;                             \
}

void write_raster(
    MaskedRasterHandle const& raster,
    std::string const& name,
    std::string const& format)
{
    // Obtain driver.
    auto driver = GetGDALDriverManager()->GetDriverByName(format.c_str());
    // TODO
    assert(driver);

    // Create new dataset, possibly overwriting existing one.
    auto metadata = driver->GetMetadata();
    // TODO
    assert(CSLFetchBoolean(metadata, GDAL_DCAP_CREATE, FALSE));

    SWITCH_ON_GDAL_SUPPORTED_VALUE_TYPE(raster->value_type(), CASE)
}

#undef CASE

} // namespace python
} // namespace fern
