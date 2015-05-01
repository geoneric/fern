// -----------------------------------------------------------------------------
// Fern © Geoneric
//
// This file is part of Geoneric Fern which is available under the terms of
// the GNU General Public License (GPL), version 2. If you do not want to
// be bound by the terms of the GPL, you may purchase a proprietary license
// from Geoneric (http://www.geoneric.eu/contact).
// -----------------------------------------------------------------------------
#include "fern/language/io/fern/fern_client.h"
#include "fern/language/io/drivers.h"
#include "fern/language/io/fern/fern_driver.h"


namespace fern {
namespace language {

size_t FernClient::_count = 0u;

std::string const FernClient::_driver_name = "Fern";


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
    drivers[_driver_name] = std::shared_ptr<Driver>(
        std::make_shared<FernDriver>());
}


void FernClient::deregister_driver()
{
    assert(drivers.find(_driver_name) != drivers.end());
    drivers.erase(_driver_name);
}

} // namespace language
} // namespace fern
