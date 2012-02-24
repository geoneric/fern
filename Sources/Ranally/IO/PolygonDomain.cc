#include "Ranally/IO/PolygonDomain.h"



namespace ranally {

PolygonDomain::PolygonDomain(
  PolygonsPtr const& polygons)

  : Domain(),
    _polygons(polygons)

{
  assert(_polygons);
}



PolygonDomain::~PolygonDomain()
{
}

} // namespace ranally

