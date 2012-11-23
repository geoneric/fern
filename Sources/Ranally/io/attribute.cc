#include "ranally/io/attribute.h"
#include "ranally/io/domain.h"
#include "ranally/io/feature.h"
#include "ranally/io/value.h"


namespace ranally {

Attribute::Attribute(
    String const& name)

    : _name(name)

{
}


Attribute::~Attribute()
{
}


String const& Attribute::name() const
{
    return _name;
}

} // namespace ranally
