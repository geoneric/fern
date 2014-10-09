#include "fern/io/core/spatial_domain.h"


namespace fern {

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

} // namespace fern
