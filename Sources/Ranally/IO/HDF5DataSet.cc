#include "Ranally/IO/HDF5DataSet.h"
#include <boost/multi_array.hpp>
#include <boost/static_assert.hpp>
#include <boost/type_traits.hpp>
#include <H5Cpp.h>
#include <hdf5.h>
#include <hdf5_hl.h>
#include "Ranally/Util/String.h"
#include "Ranally/IO/Feature.h"
#include "Ranally/IO/PointAttribute.h"
#include "Ranally/IO/PointDomain.h"
#include "Ranally/IO/PointFeature.h"
#include "Ranally/IO/PolygonAttribute.h"
#include "Ranally/IO/PolygonDomain.h"
#include "Ranally/IO/PolygonFeature.h"


namespace ranally {
namespace io {

HDF5DataSet::HDF5DataSet(
    String const& name,
    H5::H5File* file)

    : DataSet(name),
      _file(file)

{
    assert(_file);
}


HDF5DataSet::~HDF5DataSet()
{
}


herr_t incrementNrFeatures(
    hid_t /* loc_id */,
    char const* /* name */,
    H5L_info_t const* /* info */,
    void* data)
{
    ++*static_cast<size_t*>(data);
    return 0;
}


size_t HDF5DataSet::nrFeatures() const
{
    size_t nrFeatures = 0;
    herr_t result = H5Literate(_file->getLocId(), H5_INDEX_NAME, H5_ITER_NATIVE,
        NULL, incrementNrFeatures, &nrFeatures);
    if(result < 0) {
        // TODO
        H5Eprint1(stdout);
        throw std::string("error while iterating over group's links");
    }
    return nrFeatures;
}


Feature* HDF5DataSet::feature(
    size_t /* i */) const
{
    // TODO
    assert(false);
    return 0;
}


Feature* HDF5DataSet::feature(
  String const& name) const
{
    // TODO hier verder

    assert(false);
    return 0;
}


bool HDF5DataSet::exists(
    String const& name) const
{
    // Check whether group with name /name is present.
    H5G_info_t groupInfo;
    herr_t result = H5Gget_info_by_name(_file->getLocId(),
      name.encodeInUTF8().c_str(), &groupInfo, H5P_DEFAULT);
    return result >= 0;
}


//!
/*!
  \tparam    .
  \param     .
  \return    .
  \exception .
  \warning   Removing a feature from a file doesn't make the file smaller.
             Use h5repack to copy the file to a smaller version.
  \sa        .
*/
void HDF5DataSet::remove(
    String const& name)
{
    herr_t result = H5Ldelete(_file->getLocId(), name.encodeInUTF8().c_str(),
        H5P_DEFAULT);

    if(result < 0) {
        // TODO
        H5Eprint1(stdout);
        throw std::string("error while removing feature");
    }
}


template<>
void HDF5DataSet::add(
    PointFeature const& feature)
{
    // Remove feature group if it already exists.
    if(exists(feature.name())) {
        remove(feature.name());
    }

    // Create new feature group at /<feature name>.
    hid_t featureGroupId;
    {
        std::string featurePathName = feature.name().encodeInUTF8();
        featureGroupId = H5Gcreate2(_file->getLocId(),
          featurePathName.c_str(), H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        if(featureGroupId < 0) {
          // TODO
          throw std::string("error while creating group for feature");
        }
    }
    assert(featureGroupId >= 0);

    // Create domain data set at /<feature name>/__domain__.
    // The domain of a point feature contains the coordinates of points that
    // need to be written to the data set. The domain consists of a 2D matrix,
    // of dimensions nrPoints x nrCoordinates:
    // x1  | y1
    // x2  | y2
    // ... | ...
    // xn  | yn
    {
        std::string domainPathName = "__domain__";
        PointDomain const& domain(feature.domain());
        size_t const nrDimensions = 2;
        size_t const nrCoordinates = 2;
        size_t const nrPoints = domain.points().size();
        hsize_t dimensions[nrDimensions];
        dimensions[0] = nrPoints;
        dimensions[1] = nrCoordinates;
        // hid_t dataSpaceId = H5Screate_simple(2, dimensions, NULL);
        // if(dataSpaceId < 0) {
        //     // TODO
        //     throw std::string("error while creating data space");
        // }

        // Create buffer with the coordinates.
        boost::multi_array<Coordinate, 2> coordinates(
            boost::extents[nrPoints][nrCoordinates], boost::c_storage_order());
        for(size_t i = 0; i < nrPoints; ++i) {
            Point const& point(domain.points()[i]);
            coordinates[i][0] = point.get<0>();
            coordinates[i][1] = point.get<1>();
        }

        BOOST_STATIC_ASSERT(boost::is_floating_point<Coordinate>::value);
        BOOST_STATIC_ASSERT(sizeof(Coordinate) == 8);
        // hid_t dataSetId = H5Dcreate2(featureGroupId, domainPathName.c_str(),
        //     H5T_IEEE_F64LE, dataSpaceId, H5P_DEFAULT, H5P_DEFAULT,
        //     H5P_DEFAULT);

        herr_t result = H5LTmake_dataset_double(featureGroupId,
            domainPathName.c_str(), nrDimensions, dimensions,
            coordinates.data());
        if(result < 0) {
            // TODO
            throw std::string("error while creating data set");
        }
    }

    // // Create attributes at /<feature name>/<attribute name>.
    // {
    //     std::string attributePathName = ranally::util::encodeInUTF8(feature.name());
    //     hid_t featureGroupId = H5Gcreate2(_file->getLocId(),
    //         featurePathName.c_str(), H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    //     if(featureGroupId < 0) {
    //         // TODO
    //         throw std::string("error while creating group for feature");
    //     }
    // }

    // TODO
    // - Create value for each attribute at
    //   /<feature name>/<attribute name>/__value__.

    herr_t result = H5Gclose(featureGroupId);
    if(result < 0) {
        // TODO
        throw std::string("error while closing group for feature");
    }
}


template<>
void HDF5DataSet::add(
    PolygonFeature const& /* feature */)
{
    // TODO
}


void HDF5DataSet::addFeature(
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


void HDF5DataSet::copy(
    DataSet const& dataSet)
{
    for(size_t i = 0; i < dataSet.nrFeatures(); ++i) {
        boost::scoped_ptr<Feature> feature(dataSet.feature(i));
        assert(feature);
        copy(*feature);
    }
}


void HDF5DataSet::copy(
    Feature const& /* feature */)
{
}

} // namespace io
} // namespace ranally
