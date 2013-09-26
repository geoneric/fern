#include "geoneric/io/gdal/drivers.h"
#include "geoneric/io/gdal/gdal_driver.h"


namespace geoneric {

std::vector<std::shared_ptr<Driver>> drivers({
    std::shared_ptr<Driver>(new GDALDriver())
});

std::shared_ptr<Dataset> open(
    String const& name)
{
    std::shared_ptr<Dataset> dataset;

    for(auto driver: drivers) {
        dataset = driver->open(name);
        if(dataset) {
            break;
        }
    }

    if(!dataset) {
        // TODO raise exception
        assert(false);
    }

    return dataset;
}

} // namespace geoneric
