#include "fern/io/core/domain.h"


namespace fern {

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

} // namespace fern
