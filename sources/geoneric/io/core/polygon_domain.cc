#include "geoneric/io/polygon_domain.h"


namespace geoneric {

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

} // namespace geoneric
