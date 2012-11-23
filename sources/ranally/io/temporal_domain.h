#pragma once
#include "ranally/io/domain.h"


namespace ranally {

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

    bool           isSpatial           () const;

    bool           isTemporal          () const;

protected:

                   TemporalDomain      (Type type);

private:

};

} // namespace ranally
