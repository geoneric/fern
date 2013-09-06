#pragma once
#include "geoneric/io/core/geometry.h"
#include "geoneric/io/core/spatial_domain.h"


namespace geoneric {

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

} // namespace geoneric
