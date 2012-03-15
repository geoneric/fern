#ifndef INCLUDED_RANALLY_POINTDOMAIN
#define INCLUDED_RANALLY_POINTDOMAIN

#include "Ranally/IO/Geometry.h"
#include "Ranally/IO/SpatialDomain.h"



namespace ranally {

//! short_description_HORRIBLE_LONG_STRING_TO_NOTICE_THAT_IT_SHOULD_BE_REPLACED
/*!
  longer_description_HORRIBLE_LONG_STRING_TO_NOTICE_THAT_IT_SHOULD_BE_REPLACED

  \sa        .
*/
class PointDomain:
  public SpatialDomain
{

  friend class PointDomainTest;

public:

                   PointDomain         (PointsPtr const& point);

                   ~PointDomain        ();

  Points const&    points              () const;

private:

  PointsPtr        _points;

};

} // namespace ranally

#endif
