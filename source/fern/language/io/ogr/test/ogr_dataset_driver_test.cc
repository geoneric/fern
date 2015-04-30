// -----------------------------------------------------------------------------
// Fern © Geoneric
//
// This file is part of Geoneric Fern which is available under the terms of
// the GNU General Public License (GPL), version 2. If you do not want to
// be bound by the terms of the GPL, you may purchase a proprietary license
// from Geoneric (http://www.geoneric.eu/contact).
// -----------------------------------------------------------------------------
#define BOOST_TEST_MODULE fern io
#include <boost/test/unit_test.hpp>
#include "fern/language/io/ogr/ogr_client.h"
#include "fern/language/io/ogr/ogr_dataset_driver.h"
#include "fern/language/io/core/point_domain.h"
#include "fern/language/io/core/point_feature.h"


void remove_test_files()
{
    std::vector<std::string> dataset_names;
    dataset_names.emplace_back("test_create.shp");
    dataset_names.emplace_back("test_remove.shp");
    fern::OGRDatasetDriver driver("ESRI Shapefile");

    for(auto dataset_name: dataset_names) {
        if(driver.exists(dataset_name)) {
            driver.remove(dataset_name);
        }
        assert(!driver.exists(dataset_name));
    }
}


class Support:
    public fern::OGRClient
{

public:

    Support()
        : fern::OGRClient()
    {
        remove_test_files();
    }

};


BOOST_FIXTURE_TEST_SUITE(ogr_dataset_driver, Support)

BOOST_AUTO_TEST_CASE(exists)
{
    fern::OGRDatasetDriver driver("GeoJSON");

    BOOST_REQUIRE(!driver.exists("does_not_exist"));
    BOOST_REQUIRE( driver.exists("point.json"));
    BOOST_REQUIRE( driver.exists("read_only_point.json"));
    BOOST_REQUIRE(!driver.exists("write_only_point.json"));
}


BOOST_AUTO_TEST_CASE(create)
{
    fern::OGRDatasetDriver driver("ESRI Shapefile");
    std::unique_ptr<fern::OGRDataset> dataSet;

    // Test creation of new data set. ------------------------------------------
    std::string dataset_name = "test_create.shp";
    BOOST_REQUIRE(!driver.exists(dataset_name));

    dataSet.reset(driver.create(dataset_name));
    // Since there are no layers in the data set, OGR hasn't written the data
    // set to file yet.
    BOOST_CHECK(!driver.exists(dataset_name));

    // Add a feature.
    fern::PointsPtr points(std::make_shared<fern::Points>());
    points->emplace_back(fern::Point(3.0, 4.0));
    fern::PointDomainPtr domain(std::make_shared<fern::PointDomain>(points));
    fern::PointFeature feature("Stations", domain);
    dataSet->add_feature(feature);

    // Now it should work.
    BOOST_CHECK(driver.exists(dataset_name));
    BOOST_CHECK(dataSet);
    BOOST_CHECK(dataSet->name() == dataset_name);

    // Test creation of existing file. -----------------------------------------
    BOOST_CHECK(driver.exists(dataset_name));

    dataSet.reset(driver.create(dataset_name));
    // Since there are no layers in the data set, OGR hasn't written the data
    // set to file yet. The original data set should have been deleted though.
    BOOST_CHECK(!driver.exists(dataset_name));

    // The rest of the test would equal the stuff above...

    // Test creation of unwritable file. ---------------------------------------
    dataset_name = "read_only_dir/test_create.shp";
    BOOST_CHECK(!driver.exists(dataset_name));

    // This succeeds since no feature layer have been created yet.
    dataSet.reset(driver.create(dataset_name));

    // TODO Exception.
    BOOST_CHECK_THROW(dataSet->add_feature(feature), std::string);

    // Test creation of non existing path. -------------------------------------
    dataset_name = "does_not_exist/test_create.shp";
    BOOST_CHECK(!driver.exists(dataset_name));

    // This succeeds since no feature layer have been created yet.
    dataSet.reset(driver.create(dataset_name));

    // TODO Exception.
    BOOST_CHECK_THROW(dataSet->add_feature(feature), std::string);

    // TODO Test creation of Unicode path.
}


BOOST_AUTO_TEST_CASE(remove)
{
    fern::OGRDatasetDriver driver("ESRI Shapefile");
    std::unique_ptr<fern::OGRDataset> dataSet;
    std::string dataset_name;

    dataset_name = "test_remove.shp";
    BOOST_REQUIRE(!driver.exists(dataset_name));

    dataSet.reset(driver.create(dataset_name));
    BOOST_CHECK(!driver.exists(dataset_name));
    fern::PointsPtr points(std::make_shared<fern::Points>());
    points->emplace_back(fern::Point(3.0, 4.0));
    fern::PointDomainPtr domain(std::make_shared<fern::PointDomain>(points));
    fern::PointFeature feature("Stations", domain);
    dataSet->add_feature(feature);
    BOOST_CHECK(driver.exists(dataset_name));

    driver.remove(dataset_name);
    BOOST_CHECK(!driver.exists(dataset_name));

    {
        // Test remove of read-only file. --------------------------------------
        fern::OGRDatasetDriver driver("GeoJSON");
        dataset_name = "read_only_point.json";
        BOOST_CHECK(driver.exists(dataset_name));
        // TODO Read-only file is removed... Report to gdal list?!
        BOOST_WARN_THROW(driver.remove(dataset_name), std::string);

        // Test remove of non-existing file. -----------------------------------
        dataset_name = "does_not_exist.json";
        BOOST_CHECK(!driver.exists(dataset_name));
        BOOST_CHECK_THROW(driver.remove(dataset_name), std::string);
    }
}


BOOST_AUTO_TEST_CASE(open)
{
    fern::OGRDatasetDriver driver("GeoJSON");
    std::unique_ptr<fern::OGRDataset> dataSet;
    std::string dataset_name;

    dataset_name = "point.json";
    BOOST_REQUIRE(driver.exists(dataset_name));
    dataSet.reset(driver.open(dataset_name));
    BOOST_CHECK(dataSet);
    BOOST_CHECK(dataSet->name() == dataset_name);

    // Test opening non-existing data set. -------------------------------------
    BOOST_CHECK_THROW(driver.open("does_not_exist.json"), std::string);

    // Test opening read-only data set. ----------------------------------------
    BOOST_CHECK_THROW(driver.open("read_only_point.json"), std::string);
    BOOST_REQUIRE(driver.exists(dataset_name));
    dataSet.reset(driver.open(dataset_name));
    BOOST_CHECK(dataSet);
    BOOST_CHECK(dataSet->name() == dataset_name);

    // Test opening write-only data set. ---------------------------------------
    BOOST_CHECK_THROW(driver.open("write_only_point.json"), std::string);

    // Test opening executable-only data set. ----------------------------------
    BOOST_CHECK_THROW(driver.open("executable_only_point.json"), std::string);
}

BOOST_AUTO_TEST_SUITE_END()
