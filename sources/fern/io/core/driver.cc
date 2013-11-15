#include "fern/io/core/driver.h"


namespace fern {

Driver::Driver(
    String const& name)

    : _name(name)

{
}


String const& Driver::name() const
{
    return _name;
}

} // namespace fern
