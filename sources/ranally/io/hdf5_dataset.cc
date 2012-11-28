#include "ranally/io/hdf5_dataset.h"
#include <memory>
#include <type_traits>
#include <boost/multi_array.hpp>
#include <cpp/H5Cpp.h>
#include <hdf5.h>
#include <hdf5_hl.h>
#include "ranally/core/string.h"
#include "ranally/io/feature.h"
#include "ranally/io/point_attribute.h"
#include "ranally/io/point_domain.h"
#include "ranally/io/point_feature.h"
#include "ranally/io/polygon_attribute.h"
#include "ranally/io/polygon_domain.h"
#include "ranally/io/polygon_feature.h"


namespace ranally {

HDF5Dataset::HDF5Dataset(
    String const& name,
    H5::H5File* file)

    : Dataset(name),
      _file(file)

{
    assert(_file);
}


HDF5Dataset::~HDF5Dataset()
{
}


herr_t increment_nr_features(
    hid_t /* loc_id */,
    char const* /* name */,
    H5L_info_t const* /* info */,
    void* data)
{
    ++*static_cast<size_t*>(data);
    return 0;
}


size_t HDF5Dataset::nr_features() const
{
    size_t nr_features = 0;
    herr_t result = H5Literate(_file->getLocId(), H5_INDEX_NAME, H5_ITER_NATIVE,
        NULL, increment_nr_features, &nr_features);
    if(result < 0) {
        // TODO
        H5Eprint1(stdout);
        throw std::string("error while iterating over group's links");
    }
    return nr_features;
}


Feature* HDF5Dataset::feature(
    size_t /* i */) const
{
    // TODO
    assert(false);
    return 0;
}


Feature* HDF5Dataset::feature(
  String const& /* name */) const
{
    // TODO hier verder

    assert(false);
    return 0;
}


bool HDF5Dataset::exists(
    String const& name) const
{
    // Check whether group with name /name is present.
    H5G_info_t group_info;
    herr_t result = H5Gget_info_by_name(_file->getLocId(),
      name.encode_in_utf8().c_str(), &group_info, H5P_DEFAULT);
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
void HDF5Dataset::remove(
    String const& name)
{
    herr_t result = H5Ldelete(_file->getLocId(), name.encode_in_utf8().c_str(),
        H5P_DEFAULT);

    if(result < 0) {
        // TODO
        H5Eprint1(stdout);
        throw std::string("error while removing feature");
    }
}


template<>
void HDF5Dataset::add(
    PointFeature const& feature)
{
    // Remove feature group if it already exists.
    if(exists(feature.name())) {
        remove(feature.name());
    }

    // Create new feature group at /<feature name>.
    hid_t feature_group_id;
    {
        std::string feature_pathname = feature.name().encode_in_utf8();
        feature_group_id = H5Gcreate2(_file->getLocId(),
          feature_pathname.c_str(), H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        if(feature_group_id < 0) {
          // TODO
          throw std::string("error while creating group for feature");
        }
    }
    assert(feature_group_id >= 0);

    // Create domain data set at /<feature name>/__domain__.
    // The domain of a point feature contains the coordinates of points that
    // need to be written to the data set. The domain consists of a 2D matrix,
    // of dimensions nr_points x nr_coordinates:
    // x1  | y1
    // x2  | y2
    // ... | ...
    // xn  | yn
    {
        std::string domain_pathname = "__domain__";
        PointDomain const& domain(feature.domain());
        size_t const nr_dimensions = 2;
        size_t const nr_coordinates = 2;
        size_t const nr_points = domain.points().size();
        hsize_t dimensions[nr_dimensions];
        dimensions[0] = nr_points;
        dimensions[1] = nr_coordinates;
        // hid_t dataSpaceId = H5Screate_simple(2, dimensions, NULL);
        // if(dataSpaceId < 0) {
        //     // TODO
        //     throw std::string("error while creating data space");
        // }

        // Create buffer with the coordinates.
        boost::multi_array<Coordinate, 2> coordinates(
            boost::extents[nr_points][nr_coordinates],
            boost::c_storage_order());
        for(size_t i = 0; i < nr_points; ++i) {
            Point const& point(domain.points()[i]);
            coordinates[i][0] = point.get<0>();
            coordinates[i][1] = point.get<1>();
        }

        static_assert(std::is_floating_point<Coordinate>::value,
            "Coordinate must be a floating point type");
        static_assert(sizeof(Coordinate) == 8,
            "Size of Coordinate must be 8 bytes");
        // hid_t dataSetId = H5Dcreate2(feature_group_id, domain_pathname.c_str(),
        //     H5T_IEEE_F64LE, dataSpaceId, H5P_DEFAULT, H5P_DEFAULT,
        //     H5P_DEFAULT);

        herr_t result = H5LTmake_dataset_double(feature_group_id,
            domain_pathname.c_str(), nr_dimensions, dimensions,
            coordinates.data());
        if(result < 0) {
            // TODO
            throw std::string("error while creating data set");
        }
    }

    // // Create attributes at /<feature name>/<attribute name>.
    // {
    //     std::string attributePathName = ranally::util::encode_in_utf8(feature.name());
    //     hid_t feature_group_id = H5Gcreate2(_file->getLocId(),
    //         feature_pathname.c_str(), H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    //     if(feature_group_id < 0) {
    //         // TODO
    //         throw std::string("error while creating group for feature");
    //     }
    // }

    // TODO
    // - Create value for each attribute at
    //   /<feature name>/<attribute name>/__value__.

    herr_t result = H5Gclose(feature_group_id);
    if(result < 0) {
        // TODO
        throw std::string("error while closing group for feature");
    }
}


template<>
void HDF5Dataset::add(
    PolygonFeature const& /* feature */)
{
    // TODO
}


void HDF5Dataset::add_feature(
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


void HDF5Dataset::copy(
    Dataset const& dataSet)
{
    for(size_t i = 0; i < dataSet.nr_features(); ++i) {
        std::unique_ptr<Feature> feature(dataSet.feature(i));
        assert(feature);
        copy(*feature);
    }
}


void HDF5Dataset::copy(
    Feature const& /* feature */)
{
}

} // namespace ranally
