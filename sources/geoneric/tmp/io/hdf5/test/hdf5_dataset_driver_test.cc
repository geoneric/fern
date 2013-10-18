#define BOOST_TEST_MODULE geoneric io
#include <boost/test/unit_test.hpp>
#include "geoneric/core/string.h"
#include "geoneric/io/hdf5/hdf5_client.h"
#include "geoneric/io/hdf5/hdf5_dataset_driver.h"
#include "geoneric/io/core/point_attribute.h"
#include "geoneric/io/core/point_domain.h"
#include "geoneric/io/core/point_feature.h"


void remove_test_files()
{
    std::vector<geoneric::String> dataset_names;
    dataset_names.push_back("test_exists.h5");
    dataset_names.push_back("test_create.h5");
    dataset_names.push_back("test_remove.h5");
    dataset_names.push_back("test_open.h5");
    geoneric::HDF5DatasetDriver driver;

    for(auto dataset_name: dataset_names) {
        if(driver.exists(dataset_name)) {
            driver.remove(dataset_name);
        }
        assert(!driver.exists(dataset_name));
    }
}


class Support:
    public geoneric::HDF5Client
{

public:

    Support()
        : geoneric::HDF5Client()
    {
        remove_test_files();
    }

};


BOOST_FIXTURE_TEST_SUITE(hdf5_dataset_driver, Support)

BOOST_AUTO_TEST_CASE(exists)
{
    geoneric::HDF5DatasetDriver driver;

    BOOST_REQUIRE(!driver.exists("test_exists.h5"));
    std::unique_ptr<geoneric::HDF5Dataset>(driver.create("test_exists.h5"));
    BOOST_CHECK(driver.exists("test_exists.h5"));
}


BOOST_AUTO_TEST_CASE(create)
{
    // TODO Crashes.
    return;

    geoneric::HDF5DatasetDriver driver;
    geoneric::String dataset_name = "test_create.h5";
    BOOST_REQUIRE(!driver.exists(dataset_name));
    std::unique_ptr<geoneric::HDF5Dataset> dataset;

    // Create empty data set.
    {
        dataset.reset();
        dataset.reset(driver.create(dataset_name));
        BOOST_CHECK(driver.exists(dataset_name));

        dataset.reset();
        dataset.reset(driver.open(dataset_name));

        BOOST_CHECK(dataset);
        BOOST_CHECK(dataset->name() == dataset_name);
        BOOST_CHECK_EQUAL(dataset->nr_features(), 0u);
    }

    // Create a data set with a feature without attributes.
    {
        geoneric::PointsPtr points(new geoneric::Points);
        points->push_back(geoneric::Point(3.0, 4.0));
        geoneric::PointDomainPtr domain(new geoneric::PointDomain(points));
        geoneric::PointFeature feature_written("Stations", domain);

        dataset.reset();
        dataset.reset(driver.create(dataset_name));
        dataset->add_feature(feature_written);

        dataset.reset();
        dataset.reset(driver.open(dataset_name));
        BOOST_CHECK_EQUAL(dataset->nr_features(), 1u);
        BOOST_CHECK(dataset->exists("Stations"));

        geoneric::PointFeaturePtr feature_read(
            dynamic_cast<geoneric::PointFeature*>(dataset->feature("Stations")));
        // TODO BOOST_CHECK(*feature_read == feature_written);
        // BOOST_CHECK_EQUAL(feature_read->attributes().size(), 0u);
    }

    // Add a feature with an attribute.
    {
        geoneric::PointsPtr points(new geoneric::Points);
        points->push_back(geoneric::Point(3.0, 4.0));
        geoneric::PointDomainPtr domain(new geoneric::PointDomain(points));;
        geoneric::PointAttributePtr attribute(new geoneric::PointAttribute(
          "Measuring", domain));
        geoneric::PointFeature feature_written("Stations", domain);
        feature_written.add(attribute);

        dataset.reset();
        dataset.reset(driver.create(dataset_name));
        dataset->add_feature(feature_written);

        dataset.reset();
        dataset.reset(driver.open(dataset_name));
        BOOST_CHECK_EQUAL(dataset->nr_features(), 1u);
        BOOST_CHECK(dataset->exists("Stations"));

        geoneric::PointFeaturePtr feature_read(
            dynamic_cast<geoneric::PointFeature*>(dataset->feature("Stations")));
        // TODO BOOST_CHECK(*feature_read == feature_written);
        // BOOST_CHECK_EQUAL(feature_read->attributes().size(), 1u);
        // BOOST_CHECK(feature_read->exists("Measuring"));
    }


    // TODO Test creation of existing file.
    // TODO Test creation of unwritable file.
    // TODO Test creation of non existing path.
    // TODO Test creation of Unicode path.
}


BOOST_AUTO_TEST_CASE(remove)
{
    geoneric::HDF5DatasetDriver driver;
    geoneric::String dataset_name = "test_remove.h5";
    BOOST_REQUIRE(!driver.exists(dataset_name));

    std::unique_ptr<geoneric::HDF5Dataset>(driver.create(dataset_name));
    BOOST_CHECK(driver.exists(dataset_name));

    driver.remove(dataset_name);
    BOOST_CHECK(!driver.exists(dataset_name));

    // TODO Test remove of read-only file.
    // TODO Test remove of non-existing file.
}


BOOST_AUTO_TEST_CASE(open)
{
    geoneric::HDF5DatasetDriver driver;
    geoneric::String dataset_name = "test_open.h5";
    BOOST_REQUIRE(!driver.exists(dataset_name));

    std::unique_ptr<geoneric::HDF5Dataset>(driver.create(dataset_name));
    BOOST_REQUIRE(driver.exists(dataset_name));

    std::unique_ptr<geoneric::HDF5Dataset> dataset(driver.open(
      dataset_name));
    BOOST_CHECK(dataset);
    BOOST_CHECK(dataset->name() == dataset_name);

    // TODO Test opening non-existing data set.
    // TODO Test opening read-only data set.
    // TODO Test opening write-only data set.
    // TODO Test opening executable-only data set.
}

BOOST_AUTO_TEST_SUITE_END()
