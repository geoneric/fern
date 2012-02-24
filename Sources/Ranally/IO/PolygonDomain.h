#ifndef INCLUDED_RANALLY_POLYGONDOMAIN
#define INCLUDED_RANALLY_POLYGONDOMAIN

#include <boost/geometry/core/cs.hpp> 
#include <boost/geometry/geometries/point.hpp>
#include <boost/geometry/geometries/polygon.hpp>
#include "Ranally/IO/Domain.h"



namespace ranally {

//! short_description_HORRIBLE_LONG_STRING_TO_NOTICE_THAT_IT_SHOULD_BE_REPLACED
/*!
  longer_description_HORRIBLE_LONG_STRING_TO_NOTICE_THAT_IT_SHOULD_BE_REPLACED

  \sa        .
*/
class PolygonDomain:
  public Domain
{

  friend class PolygonDomainTest;

public:

                   PolygonDomain       ();

                   ~PolygonDomain      ();

private:

  //! Polygons that define the spatial domain for the attribute values.
  std::vector<boost::geometry::model::polygon<boost::geometry::model::point<
    double, 2, boost::geometry::cs::cartesian > > > _polygons;

  // TODO What about the coordinate system? How to go about that? Can we pick
  //      one a project all incoming data to that cs?

};

} // namespace ranally

#endif
