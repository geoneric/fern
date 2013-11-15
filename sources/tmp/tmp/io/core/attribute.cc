#include "fern/io/core/attribute.h"
#include "fern/io/core/domain.h"
#include "fern/io/core/feature.h"
#include "fern/io/core/value.h"


namespace fern {

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

} // namespace fern
