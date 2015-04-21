// -----------------------------------------------------------------------------
// Fern © Geoneric
//
// This file is part of Geoneric Fern which is available under the terms of
// the GNU General Public License (GPL), version 2. If you do not want to
// be bound by the terms of the GPL, you may purchase a proprietary license
// from Geoneric (http://www.geoneric.eu/contact).
// -----------------------------------------------------------------------------
#include "fern/io/gdal/dataset.h"
#include "fern/core/io_error.h"
#include "fern/io/core/file.h"


namespace fern {
namespace io {
namespace gdal {

/*!
    @ingroup    fern_io_gdal_group
    @brief      Open dataset @a name in @a mode.
    @param      mode Either GA_ReadOnly or GA_Update.
    @exception  IOError In case @a name cannot be opened in @a mode.
*/
DatasetHandle open_dataset(
    std::string const& name,
    GDALAccess mode)
{
    // This assumes name is the pathname of a file.
    if(!file_exists(name)) {
        throw IOError(name,
            Exception::messages()[MessageId::DOES_NOT_EXIST]);
    }

    // Open raster dataset.
    auto dataset = static_cast<GDALDataset *>(GDALOpen(name.c_str(), mode));

    if(dataset == nullptr) {
        assert(mode == GA_ReadOnly || mode == GA_Update);
        throw IOError(name, Exception::messages()[
            mode == GA_ReadOnly
                ? MessageId::CANNOT_BE_READ
                : MessageId::CANNOT_BE_WRITTEN]);
    }

    return DatasetHandle(dataset);
}


/*!
    @ingroup    fern_io_gdal_group
    @brief      Open dataset @a data_name in @a mode.
    @sa         open_dataset(std::string const&, GDALAccess)
*/
DatasetHandle open_dataset(
    DataName const& data_name,
    GDALAccess mode)
{
    std::string pathname{data_name.database_pathname().native_string()};

    return open_dataset(pathname, mode);
}


/*!
    @ingroup    fern_io_gdal_group
    @brief      Return the GDAL raster band @a band_id from @a dataset.
    @param      band_id Id of the raster band to return.
    @todo       Raise IOError in case @a dataset does not contain a raster
                band with id @a band_id.
*/
GDALRasterBand* raster_band(
    DatasetHandle& dataset,
    int band_id)
{
    assert(band_id > 0);

    int nr_bands{dataset->GetRasterCount()};

    if(band_id > nr_bands) {
        assert(false);
    }

    GDALRasterBand* band{dataset->GetRasterBand(band_id)};
    assert(band != nullptr);

    return band;
}


/*!
    @ingroup    fern_io_gdal_group
    @brief      Return the GDAL value type of dataset @a data_name.
    @sa         open_dataset(DataName const&, GDALAccess)

    The value type of the first band is returned.
*/
GDALDataType gdal_value_type(
    DataName const& data_name)
{
    auto dataset = open_dataset(data_name, GA_ReadOnly);
    auto band = raster_band(dataset, 1);

    return band->GetRasterDataType();
}

} // namespace gdal
} // namespace io
} // namespace fern
