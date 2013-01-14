#include "ranally/feature/attribute.h"
#include <cassert>


namespace ranally {

Attribute::Attribute(
    String const& name)

    : _name(name)

{
    assert(!name.is_empty());
}


// Attribute::~Attribute()
// {
// }


String const& Attribute::name() const
{
    return _name;
}

} // namespace ranally
