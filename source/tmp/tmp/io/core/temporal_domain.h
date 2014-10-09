#pragma once
#include "fern/io/core/domain.h"


namespace fern {

//! short_description_HORRIBLE_LONG_STRING_TO_NOTICE_THAT_IT_SHOULD_BE_REPLACED
/*!
  longer_description_HORRIBLE_LONG_STRING_TO_NOTICE_THAT_IT_SHOULD_BE_REPLACED

  \sa        .
*/
class TemporalDomain:
    public Domain
{

    friend class TemporalDomainTest;

public:

    virtual        ~TemporalDomain     ();

    bool           is_spatial          () const;

    bool           is_temporal         () const;

protected:

                   TemporalDomain      (Type type);

private:

};

} // namespace fern