#include "Ranally/IO/SpatialDomain.h"


namespace ranally {

SpatialDomain::SpatialDomain(
    Type type)

    : Domain(type)

{
}


SpatialDomain::~SpatialDomain()
{
}


bool SpatialDomain::isSpatial() const
{
    return true;
}


bool SpatialDomain::isTemporal() const
{
    return false;
}

} // namespace ranally
