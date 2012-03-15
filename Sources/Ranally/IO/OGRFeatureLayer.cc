#include "Ranally/IO/OGRFeatureLayer.h"
#include "ogrsf_frmts.h"
#include "Ranally/IO/PointDomain.h"
#include "Ranally/IO/PolygonDomain.h"
#include "Ranally/Util/String.h"



namespace ranally {
namespace io {
namespace {

Domain::Type domainType(
  OGRwkbGeometryType geometryType)
{
  Domain::Type domainType;

  switch(geometryType) {
    case wkbPoint: {
      domainType = Domain::PointDomain;
    }
    case wkbPolygon: {
      domainType = Domain::PolygonDomain;
    }
#ifndef NDEBUG
    default: {
      assert(false);
    }
#endif
  }

  return domainType;
}

} // Anonymous namespace.



OGRFeatureLayer::OGRFeatureLayer(
  OGRLayer* const layer)

  : _layer(layer)

{
  switch(domainType(_layer->GetGeomType())) {
    case Domain::PointDomain: {
      PointsPtr points;
      _domain.reset(new PointDomain(points));
    }
    case Domain::PolygonDomain: {
      PolygonsPtr polygons;
      _domain.reset(new PolygonDomain(polygons));
    }
#ifndef NDEBUG
    default: {
      assert(false);
    }
#endif
  }

  assert(_domain);
}



OGRFeatureLayer::~OGRFeatureLayer()
{
}



UnicodeString OGRFeatureLayer::name() const
{
  return util::decodeFromUTF8(_layer->GetName());
}



Domain const& OGRFeatureLayer::domain() const
{
  assert(_domain);
  return *_domain;
}

} // namespace io
} // namespace ranally

