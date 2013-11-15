#include "fern/io/fern/fern_client.h"
#include "fern/io/drivers.h"
#include "fern/io/fern/fern_driver.h"


namespace fern {

size_t FernClient::_count = 0u;

String const FernClient::_driver_name = "Fern";


FernClient::FernClient()
{
    ++_count;

    if(_count == 1u) {
        register_driver();
    }
}


FernClient::~FernClient()
{
    assert(_count > 0u);
    --_count;

    if(_count == 0u) {
        deregister_driver();
    }
}


void FernClient::register_driver()
{
    assert(drivers.find(_driver_name) == drivers.end());
    drivers[_driver_name] = std::shared_ptr<Driver>(new FernDriver());
}


void FernClient::deregister_driver()
{
    assert(drivers.find(_driver_name) != drivers.end());
    drivers.erase(_driver_name);
}

} // namespace fern
