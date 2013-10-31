#define BOOST_TEST_MODULE geoneric io
#include <boost/test/unit_test.hpp>
#include "geoneric/core/io_error.h"
#include "geoneric/feature/core/attributes.h"
#include "geoneric/io/geoneric/geoneric_dataset.h"
#include "geoneric/io/geoneric/hdf5_client.h"


class Support:
    private geoneric::HDF5Client
{

public:

    // Support()
    // {
    //     H5::Exception::dontPrint();
    //     H5::H5Library::open();
    // }

    // ~Support()
    // {
    //     H5::H5Library::close();
    // }

};


BOOST_FIXTURE_TEST_SUITE(geoneric_dataset, Support)

BOOST_AUTO_TEST_CASE(write_and_read)
{
    // Create a new dataset.
    std::shared_ptr<geoneric::GeonericDataset> dataset(
        std::make_shared<geoneric::GeonericDataset>("dataset_test_write.gnr",
            geoneric::OpenMode::OVERWRITE));
    BOOST_CHECK_EQUAL(dataset->nr_features(), 0u);

    {
        geoneric::ConstantAttribute<int32_t> constant(5);
        dataset->write_attribute(constant, "planets/gravity");
        BOOST_CHECK_EQUAL(dataset->nr_features(), 1u);
        BOOST_CHECK_EQUAL(dataset->nr_features("planets"), 0u);
        BOOST_CHECK_EQUAL(dataset->nr_attributes("planets"), 1u);
        BOOST_CHECK(dataset->contains_feature("planets"));
        BOOST_CHECK(dataset->contains_attribute("planets/gravity"));
    }

    // Read attribute without re-opening the dataset.
    {
        std::shared_ptr<geoneric::Attribute> attribute(dataset->read_attribute(
            "planets/gravity"));
        BOOST_REQUIRE(attribute);
        std::shared_ptr<geoneric::ConstantAttribute<int32_t>>
            constant_attribute(
                std::dynamic_pointer_cast<geoneric::ConstantAttribute<int32_t>>(
                    attribute));
        BOOST_REQUIRE(constant_attribute);
        BOOST_CHECK_EQUAL(constant_attribute->values().value(), 5);
    }

    // Read attribute after the dataset was closed.
    dataset = std::make_shared<geoneric::GeonericDataset>(
        "dataset_test_write.gnr", geoneric::OpenMode::READ);
    {
        std::shared_ptr<geoneric::Attribute> attribute(dataset->read_attribute(
            "planets/gravity"));
        BOOST_REQUIRE(attribute);
        std::shared_ptr<geoneric::ConstantAttribute<int32_t>>
            constant_attribute(
                std::dynamic_pointer_cast<geoneric::ConstantAttribute<int32_t>>(
                    attribute));
        BOOST_REQUIRE(constant_attribute);
        BOOST_CHECK_EQUAL(constant_attribute->values().value(), 5);
    }

    // Update the value.
    dataset = std::make_shared<geoneric::GeonericDataset>(
        "dataset_test_write.gnr", geoneric::OpenMode::UPDATE);
    {
        geoneric::ConstantAttribute<int32_t> constant(6);
        dataset->write_attribute(constant, "planets/gravity");
        std::shared_ptr<geoneric::Attribute> attribute(dataset->read_attribute(
            "planets/gravity"));
        BOOST_REQUIRE(attribute);
        std::shared_ptr<geoneric::ConstantAttribute<int32_t>>
            constant_attribute(
                std::dynamic_pointer_cast<geoneric::ConstantAttribute<int32_t>>(
                    attribute));
        BOOST_REQUIRE(constant_attribute);
        BOOST_CHECK_EQUAL(constant_attribute->values().value(), 6);
    }
}


BOOST_AUTO_TEST_CASE(errors)
{
    using namespace geoneric;

    {
        GeonericDataset dataset("constant-1.gnr", OpenMode::READ);

        try {
            dataset.read_feature("blahdiblah");
            BOOST_CHECK(false);
        }
        catch(IOError const& exception) {
            String message = exception.message();
            BOOST_CHECK_EQUAL(message,
                "IO error handling constant-1.gnr: "
                "Does not contain feature: blahdiblah");
        }

        try {
            dataset.read_attribute("blahdiblah");
            BOOST_CHECK(false);
        }
        catch(IOError const& exception) {
            String message = exception.message();
            BOOST_CHECK_EQUAL(message,
                "IO error handling constant-1.gnr: "
                "Does not contain attribute: blahdiblah");
        }
    }
}

BOOST_AUTO_TEST_SUITE_END()
