#pragma once
#include "geoneric/io/core/domain.h"


namespace geoneric {

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

    bool           is_spatial          () const;

    bool           is_temporal         () const;

protected:

                   SpatioTemporalDomain(Type type);

private:

};

} // namespace geoneric
