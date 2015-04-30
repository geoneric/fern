// -----------------------------------------------------------------------------
// Fern © Geoneric
//
// This file is part of Geoneric Fern which is available under the terms of
// the GNU General Public License (GPL), version 2. If you do not want to
// be bound by the terms of the GPL, you may purchase a proprietary license
// from Geoneric (http://www.geoneric.eu/contact).
// -----------------------------------------------------------------------------
#include "fern/language/io/ogr/ogr_feature_layer.h"
#include <ogrsf_frmts.h>
#include "fern/language/io/gdal/point_domain.h"
#include "fern/language/io/gdal/polygon_domain.h"


namespace fern {
namespace {

Domain::Type domain_type(
    OGRwkbGeometryType geometryType)
{
    Domain::Type domain_type;

    switch(geometryType) {
        case wkbPoint: {
            domain_type = Domain::PointDomain;
        }
        case wkbPolygon: {
            domain_type = Domain::PolygonDomain;
        }
#ifndef NDEBUG
        default: {
            assert(false);
        }
#endif
    }

    return domain_type;
}

} // Anonymous namespace.


OGRFeatureLayer::OGRFeatureLayer(
    OGRLayer* const layer)

    : _layer(layer),
      _domain()

{
    switch(domain_type(_layer->GetGeomType())) {
        case Domain::PointDomain: {
            PointsPtr points;
            _domain = std::make_shared<PointDomain>(points);
        }
        case Domain::PolygonDomain: {
            PolygonsPtr polygons;
            _domain = std::make_shared<PolygonDomain>(polygons);
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


std::string OGRFeatureLayer::name() const
{
    return _layer->GetName();
}


Domain const& OGRFeatureLayer::domain() const
{
    assert(_domain);
    return *_domain;
}

} // namespace fern
