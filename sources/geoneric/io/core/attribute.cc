#include "geoneric/io/attribute.h"
#include "geoneric/io/domain.h"
#include "geoneric/io/feature.h"
#include "geoneric/io/value.h"


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
