#include "fern/io/core/polygon_domain.h"


namespace fern {

PolygonDomain::PolygonDomain(
    PolygonsPtr const& polygons)

    : SpatialDomain(Domain::PolygonDomain),
      _polygons(polygons)

{
    assert(_polygons);
}


PolygonDomain::~PolygonDomain()
{
}

} // namespace fern
