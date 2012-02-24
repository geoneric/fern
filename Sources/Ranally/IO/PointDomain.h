#ifndef INCLUDED_RANALLY_POINTDOMAIN
#define INCLUDED_RANALLY_POINTDOMAIN

#include "Ranally/IO/Domain.h"
#include "Ranally/IO/Geometry.h"



namespace ranally {

//! short_description_HORRIBLE_LONG_STRING_TO_NOTICE_THAT_IT_SHOULD_BE_REPLACED
/*!
  longer_description_HORRIBLE_LONG_STRING_TO_NOTICE_THAT_IT_SHOULD_BE_REPLACED

  \sa        .
*/
class PointDomain:
  public Domain
{

  friend class PointDomainTest;

public:

                   PointDomain         (PointsPtr const& point);

                   ~PointDomain        ();

private:

  PointsPtr        _points;

};

} // namespace ranally

#endif
