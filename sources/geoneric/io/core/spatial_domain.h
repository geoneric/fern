#pragma once
#include "geoneric/io/domain.h"


namespace geoneric {

//! short_description_HORRIBLE_LONG_STRING_TO_NOTICE_THAT_IT_SHOULD_BE_REPLACED
/*!
  longer_description_HORRIBLE_LONG_STRING_TO_NOTICE_THAT_IT_SHOULD_BE_REPLACED

  \sa        .
*/
class SpatialDomain:
    public Domain
{

    friend class SpatialDomainTest;

public:

    virtual        ~SpatialDomain      ();

    bool           is_spatial          () const;

    bool           is_temporal         () const;

protected:

                   SpatialDomain       (Type type);

private:

};

} // namespace geoneric
