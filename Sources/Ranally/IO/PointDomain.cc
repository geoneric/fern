#include "Ranally/IO/PointDomain.h"



namespace ranally {

PointDomain::PointDomain(
  PointsPtr const& points)

  : Domain(),
    _points(points)

{
  assert(_points);
}



PointDomain::~PointDomain()
{
}

} // namespace ranally

