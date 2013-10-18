#include "geoneric/io/core/point_domain.h"


namespace geoneric {

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


Points const& PointDomain::points() const
{
    return *_points;
}

} // namespace geoneric
