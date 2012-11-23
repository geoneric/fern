#include "Ranally/IO/attribute.h"
#include "Ranally/IO/domain.h"
#include "Ranally/IO/feature.h"
#include "Ranally/IO/value.h"


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
