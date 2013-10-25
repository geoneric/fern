#include "geoneric/io/core/driver.h"


namespace geoneric {

Driver::Driver(
    String const& name)

    : _name(name)

{
}


String const& Driver::name() const
{
    return _name;
}

} // namespace geoneric
