#pragma once

#include <map>
#include <memory>
#include "geoneric/io/gdal/domain.h"


namespace geoneric {

//! short_description_HORRIBLE_LONG_STRING_TO_NOTICE_THAT_IT_SHOULD_BE_REPLACED
/*!
  longer_description_HORRIBLE_LONG_STRING_TO_NOTICE_THAT_IT_SHOULD_BE_REPLACED

  \sa        .
*/
template<
    class Geometry_>
class SpatialDomain: public Domain
{

public:

    typedef Geometry_ Geometry;

    typedef size_t GID;

                   SpatialDomain       ()=default;

                   SpatialDomain       (SpatialDomain const&)=delete;

    SpatialDomain& operator=           (SpatialDomain const&)=delete;

                   SpatialDomain       (SpatialDomain&&)=delete;

    SpatialDomain& operator=           (SpatialDomain&&)=delete;

                   ~SpatialDomain      ()=default;

    GID            add                 (Geometry_ const& geometry);

    Geometry_ const& geometry           (GID const& gid);

private:

    std::map<GID, Geometry_> _geometries;

};


template<
    class Geometry_>
typename SpatialDomain<Geometry_>::GID SpatialDomain<Geometry_>::add(
    Geometry_ const& geometry)
{
    GID gid = _geometries.size();
    _geometries[gid] = geometry;
    return gid;
}


template<
    class Geometry_>
Geometry_ const& SpatialDomain<Geometry_>::geometry(
    GID const& gid)
{
    assert(_geometries.find(gid) != _geometries.end());
    return _geometries.find(gid)->second;
}

} // namespace geoneric
