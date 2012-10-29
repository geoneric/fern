#include "Ranally/IO/TemporalDomain.h"


namespace ranally {

TemporalDomain::TemporalDomain(
    Type type)

    : Domain(type)

{
}


TemporalDomain::~TemporalDomain()
{
}


bool TemporalDomain::isSpatial() const
{
    return false;
}


bool TemporalDomain::isTemporal() const
{
    return true;
}

} // namespace ranally
