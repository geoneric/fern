// -----------------------------------------------------------------------------
// Fern © Geoneric
//
// This file is part of Geoneric Fern which is available under the terms of
// the GNU General Public License (GPL), version 2. If you do not want to
// be bound by the terms of the GPL, you may purchase a proprietary license
// from Geoneric (http://www.geoneric.eu/contact).
// -----------------------------------------------------------------------------
#include "fern/language/io/ogr/ogr_dataset.h"
#include <cassert>
#include <memory>
#include <ogrsf_frmts.h>
#include "fern/language/io/ogr/ogr_feature_layer.h"
#include "fern/language/io/core/point_attribute.h"
#include "fern/language/io/core/point_domain.h"
#include "fern/language/io/core/point_feature.h"
#include "fern/language/io/core/polygon_attribute.h"
#include "fern/language/io/core/polygon_domain.h"
#include "fern/language/io/core/polygon_feature.h"


namespace fern {

OGRDataset::OGRDataset(
    std::string const& name,
    OGRDataSource* data_source)

    : Dataset(name),
      _data_source(data_source)

{
    assert(_data_source);
}


OGRDataset::~OGRDataset()
{
    OGRDataSource::DestroyDataSource(_data_source);
}


size_t OGRDataset::nr_features() const
{
    return static_cast<size_t>(_data_source->GetLayerCount());
}


Feature* OGRDataset::feature(
    size_t i) const
{
    assert(i < nr_features());
    OGRFeatureLayer layer(_data_source->GetLayer(i));
    return feature(layer);

    // Darwin's gcc 4.2.1 doesn't accept this. It thinks the OGRFeatureLayer
    // has to be copied in the call to feature(...), and OGRFeatureLayer is not
    // copyable.
    // return feature(OGRFeatureLayer(_data_source->GetLayer(i)));
}


Feature* OGRDataset::feature(
    std::string const& name) const
{
    OGRLayer* ogr_layer = _data_source->GetLayerByName(
        name.encode_in_utf8().c_str());
    if(!ogr_layer) {
        // TODO
        throw std::string("layer does not exist");
    }

    OGRFeatureLayer layer(ogr_layer);
    return feature(layer);
}


bool OGRDataset::exists(
    std::string const& /* name */) const
{
    // TODO
    assert(false);
    return false;
}


void OGRDataset::remove(
    std::string const& /* name */)
{
    // TODO
    assert(false);
}


Feature* OGRDataset::feature(
    OGRFeatureLayer const& layer) const
{
    assert(false);
    // TODO Create OGRPointFeature with IO pointers for on the fly reading, or
    //      read and create the feature here, assuming it will all fit in
    //      memory.

    // switch on GetGeomType()

    // Domain
    // assert(layer.domain().isSpatial());

    // switch(layer.domain

    Feature* feature = 0;

    switch(layer.domain().type()) {
        case Domain::PointDomain: {
            feature = new PointFeature(layer.name(),
                layer.domain<PointDomain>());
        }
        case Domain::PolygonDomain: {
            feature = new PolygonFeature(layer.name(),
                layer.domain<PolygonDomain>());
        }
    }

    assert(feature);
    return feature;
}


template<>
void OGRDataset::add(
    PointFeature const& feature)
{
    PointDomain const& domain(feature.domain());
    PointAttributes const& attributes(feature.attributes());

    for(auto attribute: attributes) {
        if(attribute->feature()) {
            // TODO error/warn
            assert(false);
        }
    }

    // TODO Remove layer if it already exists?!
    Points const& points(domain.points());
    OGRLayer* ogr_layer = _data_source->CreateLayer(
        feature.name().encode_in_utf8().c_str(), NULL, wkbPoint, NULL);

    if(!ogr_layer) {
        throw std::string("cannot create ogr feature layer");
    }

    // TODO Add attributes.
    assert(attributes.empty());

    OGRPoint ogr_point;

    for(auto point: points) {
        OGRFeature* ogr_feature = OGRFeature::CreateFeature(
            ogr_layer->GetLayerDefn());
        assert(ogr_feature);

        // ogr_feature->SetField(...)

        ogr_point.setX(point.get<0>());
        ogr_point.setY(point.get<1>());
        ogr_feature->SetGeometry(&ogr_point);

        if(ogr_layer->CreateFeature(ogr_feature) != OGRERR_NONE) {
            // TODO
            assert(false);
        }

        OGRFeature::DestroyFeature(ogr_feature);
    }
}


template<>
void OGRDataset::add(
    PolygonFeature const& feature)
{
    // PolygonDomain const& domain(feature.domain());
    PolygonAttributes const& attributes(feature.attributes());

    for(auto attribute: attributes) {
        if(attribute->feature()) {
            // TODO error/warn
            assert(false);
        }
    }

    // TODO Create a feature layer.
    // TODO Write each geometry to the feature layer, including the attribute
    //      values.
}


void OGRDataset::add_feature(
    Feature const& feature)
{
    switch(feature.domain_type()) {
        case Domain::PointDomain: {
            add<PointFeature>(dynamic_cast<PointFeature const&>(feature));
            break;
        }
        case Domain::PolygonDomain: {
            add<PolygonFeature>(dynamic_cast<PolygonFeature const&>(feature));
            break;
        }
    }
}


void OGRDataset::copy(
    Dataset const& dataset)
{
    for(size_t i = 0; i < dataset.nr_features(); ++i) {
        std::unique_ptr<Feature> feature(dataset.feature(i));
        assert(feature);
        copy(*feature);
    }
}


void OGRDataset::copy(
    Feature const& /* feature */)
{
}

} // namespace fern
