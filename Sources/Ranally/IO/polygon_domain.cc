#include "Ranally/IO/polygon_domain.h"


namespace ranally {

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

} // namespace ranally
