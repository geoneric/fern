#include "geoneric/io/core/temporal_domain.h"


namespace geoneric {

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

} // namespace geoneric
