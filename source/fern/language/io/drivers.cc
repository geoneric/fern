// -----------------------------------------------------------------------------
// Fern © Geoneric
//
// This file is part of Geoneric Fern which is available under the terms of
// the GNU General Public License (GPL), version 2. If you do not want to
// be bound by the terms of the GPL, you may purchase a proprietary license
// from Geoneric (http://www.geoneric.eu/contact).
// -----------------------------------------------------------------------------
#include "fern/language/io/drivers.h"
#include "fern/core/io_error.h"
#include "fern/language/io/gdal/gdal_driver.h"


namespace fern {
namespace language {

std::map<std::string, std::shared_ptr<Driver>> drivers;


static std::vector<std::shared_ptr<Driver>> drivers_to_try(
    std::string const& name,
    std::string const& format)
{
    assert(!drivers.empty());  // Did you forget to use an IO client?
    std::vector<std::shared_ptr<Driver>> drivers;

    if(!format.empty()) {
        if(fern::language::drivers.find(format) ==
                fern::language::drivers.end()) {
            // TODO Just throw an "no such driver" exception and let the
            //      caller add info about the name.
            throw IOError(name,
                Exception::messages().format_message(MessageId::NO_SUCH_DRIVER,
                format));
        }

        drivers.emplace_back(fern::language::drivers.at(format));
    }
    else {
        // Make sure the Fern driver is added first. GDAL may otherwise
        // think it can read Fern formatted files, which it can't.
        if(fern::language::drivers.find("Fern") !=
                fern::language::drivers.end()) {
            drivers.emplace_back(fern::language::drivers.at("Fern"));
        }

        for(auto driver: fern::language::drivers) {
            if(driver.second->name() != "Fern") {
                drivers.emplace_back(driver.second);
            }
        }
    }

    return drivers;
}


std::shared_ptr<Dataset> open_dataset(
    std::string const& name,
    OpenMode open_mode,
    std::string const& format)
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
    std::string const& name,
    OpenMode open_mode,
    std::string const& format)
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
//     std::string const& name,
//     std::string const& format)
// {
//     assert(!format.is_empty());
//     auto drivers(drivers_to_try(name, format));
//     assert(drivers.size() == 1u);
// 
//     return drivers[0]->create(attribute, name);
// }

} // namespace language
} // namespace fern
