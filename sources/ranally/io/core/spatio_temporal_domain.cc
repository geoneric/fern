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


bool SpatioTemporalDomain::is_spatial() const
{
    return true;
}


bool SpatioTemporalDomain::is_temporal() const
{
    return true;
}

} // namespace ranally
