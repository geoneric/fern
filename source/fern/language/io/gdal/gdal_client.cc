// -----------------------------------------------------------------------------
// Fern © Geoneric
//
// This file is part of Geoneric Fern which is available under the terms of
// the GNU General Public License (GPL), version 2. If you do not want to
// be bound by the terms of the GPL, you may purchase a proprietary license
// from Geoneric (http://www.geoneric.eu/contact).
// -----------------------------------------------------------------------------
#include "fern/language/io/gdal/gdal_client.h"
#include <gdal_priv.h>
#include "fern/language/io/drivers.h"
#include "fern/language/io/gdal/gdal_driver.h"


namespace fern {

size_t GDALClient::_count = 0u;

std::vector<std::string> GDALClient::_names_of_drivers_to_skip;


void GDALClient::insert_names_of_drivers_to_skip()
{
    GDALDriverManager* driver_manager = GetGDALDriverManager();

    if(GDAL_VERSION_NUM == 1110100) {  // 1.11.1
        if(driver_manager->GetDriverByName("PDF")) {
            _names_of_drivers_to_skip.emplace_back("PDF");
        }
    }
}


void GDALClient::erase_driver(
    GDALDriverManager& manager,
    std::string const& name,
    ::GDALDriver* driver)
{
    assert(drivers.find(name) != drivers.end());

    // First we need to let loose of the gdal driver.
    drivers.erase(name);

    // Then gdal needs to take its hands of it.
    manager.DeregisterDriver(driver);

    // Since nobody is using the driver anymore, we can delete it.
    GDALDestroyDriver(driver);
}


void GDALClient::erase_drivers_to_skip()
{
    GDALDriverManager& driver_manager = *GetGDALDriverManager();
    ::GDALDriver* driver;

    for(auto const& driver_name: _names_of_drivers_to_skip) {
        driver = driver_manager.GetDriverByName(driver_name.c_str());
        erase_driver(driver_manager, driver_name, driver);
    }
}


size_t GDALClient::nr_drivers_to_skip()
{
    return _names_of_drivers_to_skip.size();
}


bool GDALClient::skip_driver(
    std::string const& name)
{
    return std::find(_names_of_drivers_to_skip.begin(),
        _names_of_drivers_to_skip.end(), name) !=
        _names_of_drivers_to_skip.end();
}


void GDALClient::register_all_drivers()
{
    GDALAllRegister();
    std::string driver_name;
    ::GDALDriver* driver;
    GDALDriverManager* driver_manager = GetGDALDriverManager();

    for(int i = 0; i < driver_manager->GetDriverCount(); ++i) {
        driver = driver_manager->GetDriver(i);
        driver_name = driver->GetDescription();
        assert(drivers.find(driver_name) == drivers.end());
        drivers[driver_name] = std::shared_ptr<Driver>(
            std::make_shared<GDALDriver>(driver));
    }
}


void GDALClient::deregister_all_drivers()
{
    std::string driver_name;
    ::GDALDriver* driver;
    GDALDriverManager& driver_manager = *GetGDALDriverManager();

    while(driver_manager.GetDriverCount() > 0) {
        driver = driver_manager.GetDriver(0);
        driver_name = driver->GetDescription();
        erase_driver(driver_manager, driver_name, driver);
    }

    assert(driver_manager.GetDriverCount() == 0);
}


GDALClient::GDALClient()
{
    ++_count;

    if(_count == 1u) {
        // Don't throw in case of an error.
        CPLSetErrorHandler(CPLQuietErrorHandler);

        // First register *all* available drivers.
        register_all_drivers();

        // There are drivers that are known to contains bugs. We don't need
        // them, so just get rid of them.
        // Given the drivers that are available, determine the names of the
        // drivers that should be skipped.
        insert_names_of_drivers_to_skip();

        // Erase the drivers that should be skipped.
        erase_drivers_to_skip();
    }

}


GDALClient::~GDALClient()
{
    assert(_count > 0u);
    --_count;

    if(_count == 0u) {
        deregister_all_drivers();
        _names_of_drivers_to_skip.clear();
    }
}

} // namespace fern
