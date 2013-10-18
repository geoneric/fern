#include "geoneric/io/geoneric/geoneric_client.h"
#include "geoneric/io/drivers.h"
#include "geoneric/io/geoneric/geoneric_driver.h"


namespace geoneric {

size_t GeonericClient::_count = 0u;

String const GeonericClient::_driver_name = "Geoneric";


GeonericClient::GeonericClient()
{
    ++_count;

    if(_count == 1u) {
        register_driver();
    }
}


GeonericClient::~GeonericClient()
{
    assert(_count > 0u);
    --_count;

    if(_count == 0u) {
        deregister_driver();
    }
}


void GeonericClient::register_driver()
{
    assert(drivers.find(_driver_name) == drivers.end());
    drivers[_driver_name] = std::shared_ptr<Driver>(new GeonericDriver());
}


void GeonericClient::deregister_driver()
{
    assert(drivers.find(_driver_name) != drivers.end());
    drivers.erase(_driver_name);
}

} // namespace geoneric
