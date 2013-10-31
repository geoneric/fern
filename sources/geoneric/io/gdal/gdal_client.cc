#include "geoneric/io/gdal/gdal_client.h"
#include <gdal_priv.h>
#include "geoneric/io/drivers.h"
#include "geoneric/io/gdal/gdal_driver.h"


namespace geoneric {

size_t GDALClient::_count = 0u;


GDALClient::GDALClient()
{
    ++_count;

    if(_count == 1u) {
        // Don't throw in case of an error.
        CPLSetErrorHandler(CPLQuietErrorHandler);

        register_all_drivers();
    }
}


GDALClient::~GDALClient()
{
    assert(_count > 0u);
    --_count;

    if(_count == 0u) {
        deregister_all_drivers();
    }
}


void GDALClient::register_all_drivers()
{
    GDALAllRegister();
    String driver_name;
    ::GDALDriver* driver;
    GDALDriverManager* driver_manager = GetGDALDriverManager();

    for(int i = 0; i < driver_manager->GetDriverCount(); ++i) {
        driver = driver_manager->GetDriver(i);
        driver_name = driver->GetDescription();
        assert(drivers.find(driver_name) == drivers.end());
        drivers[driver_name] = std::shared_ptr<Driver>(new GDALDriver(driver));
    }
}


void GDALClient::deregister_all_drivers()
{
    String driver_name;
    ::GDALDriver* driver;
    GDALDriverManager* driver_manager = GetGDALDriverManager();

    while(driver_manager->GetDriverCount()) {
        driver = driver_manager->GetDriver(0);
        driver_name = driver->GetDescription();
        assert(drivers.find(driver_name) != drivers.end());

        // First we need to let loose of the gdal driver.
        drivers.erase(driver_name);

        // Then gdal needs to take its hands of it.
        driver_manager->DeregisterDriver(driver);

        // Since nobody is using the driver anymore, we can delete it.
        GDALDestroyDriver(driver);
    }
}

} // namespace geoneric
