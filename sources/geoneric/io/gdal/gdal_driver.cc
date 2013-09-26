#include "geoneric/io/gdal/gdal_driver.h"
#include "gdal_priv.h"
#include "geoneric/io/gdal/gdal_dataset.h"


namespace geoneric {

GDALDriver::GDALDriver()

    : Driver()

{
    GDALAllRegister();
}


std::shared_ptr<Dataset> GDALDriver::open(
    String const& name)
{
    std::shared_ptr<Dataset> result;

    if(GDALDataset::can_open(name)) {
        result = std::shared_ptr<Dataset>(new GDALDataset(name));
    }

    return result;
}

} // namespace geoneric
