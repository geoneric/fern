#include "Ranally/IO/OGRFeatureLayer.h"
#include "ogrsf_frmts.h"
#include "Ranally/IO/Domain.h"



namespace ranally {
namespace io {
namespace {

Domain::Type domainType(
  OGRwkbGeometryType geometryType)
{
  Domain::Type domainType = Domain::UnknownDomainType;

  switch(geometryType) {
    case wkbPoint: {
      domainType = Domain::PointDomain;
    }
    default: {
      assert(false);
    }
  }

  assert(!Domain::UnknownDomainType);
  return domainType;
}

} // Anonymous namespace.



OGRFeatureLayer::OGRFeatureLayer(
  OGRLayer* layer)

  : _layer(layer)

{
  switch(domainType(_layer->GetGeomType())) {
    case Domain::PointDomain: {
    }
    default: {
      assert(false);
    }
  }

  assert(_domain);
}



OGRFeatureLayer::~OGRFeatureLayer()
{
}



Domain const& OGRFeatureLayer::domain() const
{
  assert(_domain);
  return *_domain;
}

} // namespace io
} // namespace ranally

