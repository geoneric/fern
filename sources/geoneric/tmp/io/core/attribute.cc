#include "geoneric/io/core/attribute.h"
#include "geoneric/io/core/domain.h"
#include "geoneric/io/core/feature.h"
#include "geoneric/io/core/value.h"


namespace geoneric {

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

} // namespace geoneric
