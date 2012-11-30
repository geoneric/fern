#define BOOST_TEST_MODULE ranally io
#include <boost/test/unit_test.hpp>
#include "ranally/core/string.h"
#include "ranally/io/hdf5_client.h"
#include "ranally/io/hdf5_dataset_driver.h"
#include "ranally/io/point_attribute.h"
#include "ranally/io/point_domain.h"
#include "ranally/io/point_feature.h"


void remove_test_files()
{
    std::vector<ranally::String> dataset_names;
    dataset_names.push_back("test_exists.h5");
    dataset_names.push_back("test_create.h5");
    dataset_names.push_back("test_remove.h5");
    dataset_names.push_back("test_open.h5");
    ranally::HDF5DatasetDriver driver;

    for(auto dataset_name: dataset_names) {
        if(driver.exists(dataset_name)) {
            driver.remove(dataset_name);
        }
        assert(!driver.exists(dataset_name));
    }
}


class Support:
    public ranally::HDF5Client
{

public:

    Support()
        : ranally::HDF5Client()
    {
        remove_test_files();
    }

};


BOOST_FIXTURE_TEST_SUITE(hdf5_dataset_driver, Support)

BOOST_AUTO_TEST_CASE(exists)
{
    ranally::HDF5DatasetDriver driver;

    BOOST_REQUIRE(!driver.exists("test_exists.h5"));
    std::unique_ptr<ranally::HDF5Dataset>(driver.create("test_exists.h5"));
    BOOST_CHECK(driver.exists("test_exists.h5"));
}


BOOST_AUTO_TEST_CASE(create)
{
    // TODO Crashes.
    return;

    ranally::HDF5DatasetDriver driver;
    ranally::String dataset_name = "test_create.h5";
    BOOST_REQUIRE(!driver.exists(dataset_name));
    std::unique_ptr<ranally::HDF5Dataset> dataset;

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
        ranally::PointsPtr points(new ranally::Points);
        points->push_back(ranally::Point(3.0, 4.0));
        ranally::PointDomainPtr domain(new ranally::PointDomain(points));
        ranally::PointFeature feature_written("Stations", domain);

        dataset.reset();
        dataset.reset(driver.create(dataset_name));
        dataset->add_feature(feature_written);

        dataset.reset();
        dataset.reset(driver.open(dataset_name));
        BOOST_CHECK_EQUAL(dataset->nr_features(), 1u);
        BOOST_CHECK(dataset->exists("Stations"));

        ranally::PointFeaturePtr feature_read(
            dynamic_cast<ranally::PointFeature*>(dataset->feature("Stations")));
        // TODO BOOST_CHECK(*feature_read == feature_written);
        // BOOST_CHECK_EQUAL(feature_read->attributes().size(), 0u);
    }

    // Add a feature with an attribute.
    {
        ranally::PointsPtr points(new ranally::Points);
        points->push_back(ranally::Point(3.0, 4.0));
        ranally::PointDomainPtr domain(new ranally::PointDomain(points));;
        ranally::PointAttributePtr attribute(new ranally::PointAttribute(
          "Measuring", domain));
        ranally::PointFeature feature_written("Stations", domain);
        feature_written.add(attribute);

        dataset.reset();
        dataset.reset(driver.create(dataset_name));
        dataset->add_feature(feature_written);

        dataset.reset();
        dataset.reset(driver.open(dataset_name));
        BOOST_CHECK_EQUAL(dataset->nr_features(), 1u);
        BOOST_CHECK(dataset->exists("Stations"));

        ranally::PointFeaturePtr feature_read(
            dynamic_cast<ranally::PointFeature*>(dataset->feature("Stations")));
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
    ranally::HDF5DatasetDriver driver;
    ranally::String dataset_name = "test_remove.h5";
    BOOST_REQUIRE(!driver.exists(dataset_name));

    std::unique_ptr<ranally::HDF5Dataset>(driver.create(dataset_name));
    BOOST_CHECK(driver.exists(dataset_name));

    driver.remove(dataset_name);
    BOOST_CHECK(!driver.exists(dataset_name));

    // TODO Test remove of read-only file.
    // TODO Test remove of non-existing file.
}


BOOST_AUTO_TEST_CASE(open)
{
    ranally::HDF5DatasetDriver driver;
    ranally::String dataset_name = "test_open.h5";
    BOOST_REQUIRE(!driver.exists(dataset_name));

    std::unique_ptr<ranally::HDF5Dataset>(driver.create(dataset_name));
    BOOST_REQUIRE(driver.exists(dataset_name));

    std::unique_ptr<ranally::HDF5Dataset> dataset(driver.open(
      dataset_name));
    BOOST_CHECK(dataset);
    BOOST_CHECK(dataset->name() == dataset_name);

    // TODO Test opening non-existing data set.
    // TODO Test opening read-only data set.
    // TODO Test opening write-only data set.
    // TODO Test opening executable-only data set.
}

BOOST_AUTO_TEST_SUITE_END()
