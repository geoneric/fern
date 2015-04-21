// -----------------------------------------------------------------------------
// Fern © Geoneric
//
// This file is part of Geoneric Fern which is available under the terms of
// the GNU General Public License (GPL), version 2. If you do not want to
// be bound by the terms of the GPL, you may purchase a proprietary license
// from Geoneric (http://www.geoneric.eu/contact).
// -----------------------------------------------------------------------------
#include "fern/io/gdal/gdal_client.h"
#include <cassert>
#include <gdal_priv.h>


namespace fern {
namespace io {
namespace gdal {

size_t GDALClient::_count = 0u;


/*!
    @brief      Create an instance.

    If this is the first instance created, all drivers are loaded. Also,
    the GDAL error handler is configured to not print messages on the
    standard error output stream.

    Multiple instances can be created, but only the first one will actually
    initialize GDAL.
*/
GDALClient::GDALClient()
{
    ++_count;

    if(_count == 1u) {
        CPLSetErrorHandler(CPLQuietErrorHandler);

        register_all_drivers();
    }
}


/*!
    @brief      Destruct an instance.

    If this is the last instance destructed, all drivers are unloaded
    again. All drivers are also deleted from memory.
*/
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
}


void GDALClient::deregister_all_drivers()
{
    GDALDriver* driver;
    GDALDriverManager* driver_manager = GetGDALDriverManager();

    while(driver_manager->GetDriverCount()) {
        driver = driver_manager->GetDriver(0);

        // GDAL needs to take its hands of it.
        driver_manager->DeregisterDriver(driver);

        // Since nobody is using the driver anymore, we can delete it.
        GDALDestroyDriver(driver);
    }
}

} // namespace gdal
} // namespace io
} // namespace fern
