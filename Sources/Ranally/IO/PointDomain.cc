#include "Ranally/IO/PointDomain.h"



namespace ranally {

PointDomain::PointDomain(
  PointsPtr const& points)

  : SpatialDomain(Domain::PointDomain),
    _points(points)

{
  assert(_points);
}



PointDomain::~PointDomain()
{
}

} // namespace ranally

