#include "fern/io/drivers.h"
#include "fern/core/io_error.h"
#include "fern/io/gdal/gdal_driver.h"


namespace fern {

std::map<String, std::shared_ptr<Driver>> drivers;


static std::vector<std::shared_ptr<Driver>> drivers_to_try(
    String const& name,
    String const& format)
{
    std::vector<std::shared_ptr<Driver>> drivers;

    if(!format.is_empty()) {
        if(fern::drivers.find(format) == fern::drivers.end()) {
            // TODO Just throw an "no such driver" exception and let the
            //      caller add info about the name.
            throw IOError(name,
                Exception::messages().format_message(MessageId::NO_SUCH_DRIVER,
                format));
        }

        drivers.push_back(fern::drivers.at(format));
    }
    else {
        // Make sure the Fern driver is added first. GDAL may otherwise
        // think it can read Fern formatted files, which it can't.
        if(fern::drivers.find("Fern") != fern::drivers.end()) {
            drivers.push_back(fern::drivers.at("Fern"));
        }

        for(auto driver: fern::drivers) {
            if(driver.second->name() != "Fern") {
                drivers.push_back(driver.second);
            }
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

    for(auto driver: drivers_to_try(name, format)) {
        if(driver->can_open(name, open_mode)) {
            dataset = driver->open(name, open_mode);
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

    for(auto driver: drivers_to_try(name, format)) {
        exists = driver->can_open(name, open_mode);
        if(exists) {
            break;
        }
    }

    return exists;
}


// ExpressionType expression_type(
//     DataName const& data_name)
// {
//     std::shared_ptr<Dataset> dataset = open_dataset(
//         data_name.database_pathname(), OpenMode::READ);
//     return dataset->expression_type(data_name.data_pathname());
// }


// std::shared_ptr<Driver> driver_for(
//     Path const& path,
//     OpenMode open_mode)
// {
//     std::shared_ptr<Driver> result;
// 
//     for(auto driver: drivers_to_try(path.native_string(), "")) {
//         if(driver->can_open(path, open_mode)) {
//             result = driver;
//             break;
//         }
//     }
// 
//     if(!result) {
//         throw IOError(path.native_string(),
//             Exception::messages()[MessageId::CANNOT_BE_READ]);
//     }
// 
//     return result;
// }


// std::shared_ptr<Dataset> create_dataset(
//     Attribute const& attribute,
//     String const& name,
//     String const& format)
// {
//     assert(!format.is_empty());
//     auto drivers(drivers_to_try(name, format));
//     assert(drivers.size() == 1u);
// 
//     return drivers[0]->create(attribute, name);
// }

} // namespace fern
