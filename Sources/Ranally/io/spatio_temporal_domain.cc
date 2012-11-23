#include "ranally/io/spatio_temporal_domain.h"


namespace ranally {

SpatioTemporalDomain::SpatioTemporalDomain(
    Type type)

    : Domain(type)

{
}


SpatioTemporalDomain::~SpatioTemporalDomain()
{
}


bool SpatioTemporalDomain::isSpatial() const
{
    return true;
}


bool SpatioTemporalDomain::isTemporal() const
{
    return true;
}

} // namespace ranally
