#pragma once
#include "Ranally/IO/domain.h"


namespace ranally {

//! short_description_HORRIBLE_LONG_STRING_TO_NOTICE_THAT_IT_SHOULD_BE_REPLACED
/*!
  longer_description_HORRIBLE_LONG_STRING_TO_NOTICE_THAT_IT_SHOULD_BE_REPLACED

  \sa        .
*/
class SpatioTemporalDomain:
    public Domain
{

    friend class SpatioTemporalDomainTest;

public:

    virtual        ~SpatioTemporalDomain();

    bool           isSpatial           () const;

    bool           isTemporal          () const;

protected:

                   SpatioTemporalDomain(Type type);

private:

};

} // namespace ranally
