#include "Ranally/IO/Domain.h"


namespace ranally {

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

} // namespace ranally
