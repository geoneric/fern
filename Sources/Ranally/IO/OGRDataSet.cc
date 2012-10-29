#include "Ranally/IO/OGRDataSet.h"
#include <cassert>
#include <boost/foreach.hpp>
#include <boost/scoped_ptr.hpp>
#include "ogrsf_frmts.h"
#include "Ranally/IO/OGRFeatureLayer.h"
#include "Ranally/IO/PointAttribute.h"
#include "Ranally/IO/PointDomain.h"
#include "Ranally/IO/PointFeature.h"
#include "Ranally/IO/PolygonAttribute.h"
#include "Ranally/IO/PolygonDomain.h"
#include "Ranally/IO/PolygonFeature.h"
#include "Ranally/Util/String.h"


namespace ranally {
namespace io {

OGRDataSet::OGRDataSet(
    String const& name,
    OGRDataSource* dataSource)

    : DataSet(name),
      _dataSource(dataSource)

{
    assert(_dataSource);
}


OGRDataSet::~OGRDataSet()
{
    OGRDataSource::DestroyDataSource(_dataSource);
}


size_t OGRDataSet::nrFeatures() const
{
    return static_cast<size_t>(_dataSource->GetLayerCount());
}


Feature* OGRDataSet::feature(
    size_t i) const
{
    assert(i < nrFeatures());
    OGRFeatureLayer layer(_dataSource->GetLayer(i));
    return feature(layer);

    // Darwin's gcc 4.2.1 doesn't accept this. It thinks the OGRFeatureLayer
    // has to be copied in the call to feature(...), and OGRFeatureLayer is not
    // copyable.
    // return feature(OGRFeatureLayer(_dataSource->GetLayer(i)));
}


Feature* OGRDataSet::feature(
    String const& name) const
{
    OGRLayer* ogrLayer = _dataSource->GetLayerByName(
        name.encodeInUTF8().c_str());
    if(!ogrLayer) {
        // TODO
        throw std::string("layer does not exist");
    }

    OGRFeatureLayer layer(ogrLayer);
    return feature(layer);
}


bool OGRDataSet::exists(
    String const& name) const
{
    // TODO
    assert(false);
    return false;
}


void OGRDataSet::remove(
    String const& name)
{
    // TODO
    assert(false);
}


Feature* OGRDataSet::feature(
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
void OGRDataSet::add(
    PointFeature const& feature)
{
    PointDomain const& domain(feature.domain());
    PointAttributes const& attributes(feature.attributes());

    BOOST_FOREACH(PointAttributePtr const& attribute, attributes) {
        if(attribute->feature()) {
            // TODO error/warn
            assert(false);
        }
    }

    // TODO Remove layer if it already exists?!
    Points const& points(domain.points());
    OGRLayer* ogrLayer = _dataSource->CreateLayer(
        feature.name().encodeInUTF8().c_str(), NULL, wkbPoint, NULL);

    if(!ogrLayer) {
        throw std::string("cannot create ogr feature layer");
    }

    // TODO Add attributes.
    assert(attributes.empty());

    OGRPoint ogrPoint;

    BOOST_FOREACH(Point const& point, points) {
        OGRFeature* ogrFeature = OGRFeature::CreateFeature(
            ogrLayer->GetLayerDefn());
        assert(ogrFeature);

        // ogrFeature->SetField(...)

        ogrPoint.setX(point.get<0>());
        ogrPoint.setY(point.get<1>());
        ogrFeature->SetGeometry(&ogrPoint);

        if(ogrLayer->CreateFeature(ogrFeature) != OGRERR_NONE) {
            // TODO
            assert(false);
        }

        OGRFeature::DestroyFeature(ogrFeature);
    }
}


template<>
void OGRDataSet::add(
    PolygonFeature const& feature)
{
    PolygonDomain const& domain(feature.domain());
    PolygonAttributes const& attributes(feature.attributes());

    BOOST_FOREACH(PolygonAttributePtr const& attribute, attributes) {
        if(attribute->feature()) {
            // TODO error/warn
            assert(false);
        }
    }

    // TODO Create a feature layer.
    // TODO Write each geometry to the feature layer, including the attribute
    //      values.
}


void OGRDataSet::addFeature(
    Feature const& feature)
{
    switch(feature.domainType()) {
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


void OGRDataSet::copy(
    DataSet const& dataSet)
{
    for(size_t i = 0; i < dataSet.nrFeatures(); ++i) {
        boost::scoped_ptr<Feature> feature(dataSet.feature(i));
        assert(feature);
        copy(*feature);
    }
}


void OGRDataSet::copy(
    Feature const& /* feature */)
{
}

} // namespace io
} // namespace ranally
