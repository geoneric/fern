#include "fern/io/core/spatio_temporal_domain.h"


namespace fern {

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

} // namespace fern
