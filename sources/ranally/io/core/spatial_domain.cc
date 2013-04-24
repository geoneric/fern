#include "ranally/io/spatial_domain.h"


namespace ranally {

SpatialDomain::SpatialDomain(
    Type type)

    : Domain(type)

{
}


SpatialDomain::~SpatialDomain()
{
}


bool SpatialDomain::is_spatial() const
{
    return true;
}


bool SpatialDomain::is_temporal() const
{
    return false;
}

} // namespace ranally
