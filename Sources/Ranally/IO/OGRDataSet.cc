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



namespace ranally {
namespace io {

OGRDataSet::OGRDataSet(
  UnicodeString const& name,
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
      feature = new PointFeature(layer.domain<PointDomain>());
    }
    case Domain::PolygonDomain: {
      feature = new PolygonFeature(layer.domain<PolygonDomain>());
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

  // TODO Create a feature layer.
  // TODO Write each geometry to the feature layer, including the attribute
  //      values.
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
  Feature const* feature)
{
  switch(feature->domainType()) {
    case Domain::PointDomain: {
      add<PointFeature>(dynamic_cast<PointFeature const&>(*feature));
    }
    case Domain::PolygonDomain: {
      add<PolygonFeature>(dynamic_cast<PolygonFeature const&>(*feature));
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

