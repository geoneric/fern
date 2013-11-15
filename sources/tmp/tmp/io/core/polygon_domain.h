#pragma once
#include "fern/io/core/geometry.h"
#include "fern/io/core/spatial_domain.h"


namespace fern {

//! short_description_HORRIBLE_LONG_STRING_TO_NOTICE_THAT_IT_SHOULD_BE_REPLACED
/*!
  longer_description_HORRIBLE_LONG_STRING_TO_NOTICE_THAT_IT_SHOULD_BE_REPLACED

  \sa        .
*/
class PolygonDomain:
    public SpatialDomain
{

    friend class PolygonDomainTest;

public:

                   PolygonDomain       (PolygonsPtr const& polygons);

                   ~PolygonDomain      ();

private:

    PolygonsPtr    _polygons;

};

} // namespace fern
