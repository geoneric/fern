#include "geoneric/io/gdal/drivers.h"
#include "geoneric/core/io_error.h"
#include "geoneric/io/gdal/gdal_driver.h"


namespace geoneric {

std::map<String, std::shared_ptr<Driver>> drivers;


static std::vector<std::shared_ptr<Driver>> drivers_to_try(
    String const& format)
{
    std::vector<std::shared_ptr<Driver>> drivers;

    if(!format.is_empty()) {
        if(geoneric::drivers.find(format) == geoneric::drivers.end()) {
            throw IOError(format,
                Exception::messages()[MessageId::NO_SUCH_DRIVER]);
        }

        drivers.push_back(geoneric::drivers.at(format));
    }
    else {
        for(auto driver: geoneric::drivers) {
            drivers.push_back(driver.second);
        }
    }

    return drivers;
}


std::shared_ptr<Dataset> open_dataset(
    String const& name,
    OpenMode open_mode,
    String const& format)
{
    std::shared_ptr<Dataset> dataset;

    for(auto driver: drivers_to_try(format)) {
        dataset = driver->open(name, open_mode);
        if(dataset) {
            break;
        }
    }

    if(!dataset) {
        throw IOError(name,
            Exception::messages()[
                open_mode == OpenMode::READ
                    ? MessageId::CANNOT_BE_READ
                    : MessageId::CANNOT_BE_WRITTEN]);
    }

    // TODO Cache driver by name, only when open was successful.
    //      Maybe caller should do this. In that case, the driver must also
    //      be returned.

    return dataset;
}


bool dataset_exists(
    String const& name,
    OpenMode open_mode,
    String const& format)
{
    bool exists = false;

    for(auto driver: drivers_to_try(format)) {
        exists = driver->exists(name, open_mode);
        if(exists) {
            break;
        }
    }

    return exists;
}


std::shared_ptr<Dataset> create_dataset(
    Attribute const& attribute,
    String const& name,
    String const& format)
{
    assert(!format.is_empty());
    auto drivers(drivers_to_try(format));
    assert(drivers.size() == 1u);

    return drivers[0]->create(attribute, name);
}

} // namespace geoneric
