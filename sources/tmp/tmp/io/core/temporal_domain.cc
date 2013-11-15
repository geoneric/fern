#include "fern/io/core/temporal_domain.h"


namespace fern {

TemporalDomain::TemporalDomain(
    Type type)

    : Domain(type)

{
}


TemporalDomain::~TemporalDomain()
{
}


bool TemporalDomain::is_spatial() const
{
    return false;
}


bool TemporalDomain::is_temporal() const
{
    return true;
}

} // namespace fern
