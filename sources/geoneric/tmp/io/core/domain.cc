#include "geoneric/io/core/domain.h"


namespace geoneric {

Domain::Domain(
    Type type)

    : _type(type)

{
}


Domain::~Domain()
{
}


Domain::Type Domain::type() const
{
    return _type;
}

} // namespace geoneric
