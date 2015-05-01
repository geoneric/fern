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
#include "fern/core/io_error.h"
#include "fern/language/feature/core/attributes.h"
#include "fern/language/io/fern/fern_dataset.h"
#include "fern/language/io/fern/hdf5_client.h"


namespace fl = fern::language;


class Support:
    private fl::HDF5Client
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


BOOST_FIXTURE_TEST_SUITE(fern_dataset, Support)

BOOST_AUTO_TEST_CASE(write_and_read)
{
    // Create a new dataset.
    std::shared_ptr<fl::FernDataset> dataset(
        std::make_shared<fl::FernDataset>("dataset_test_write.gnr",
            fl::OpenMode::OVERWRITE));
    BOOST_CHECK_EQUAL(dataset->nr_features(), 0u);

    {
        fl::ConstantAttribute<int32_t> constant(5);
        dataset->write_attribute(constant, "planets/gravity");
        BOOST_CHECK_EQUAL(dataset->nr_features(), 1u);
        BOOST_CHECK_EQUAL(dataset->nr_features("planets"), 0u);
        BOOST_CHECK_EQUAL(dataset->nr_attributes("planets"), 1u);
        BOOST_CHECK(dataset->contains_feature("planets"));
        BOOST_CHECK(dataset->contains_attribute("planets/gravity"));

        auto feature_names = dataset->feature_names();
        BOOST_REQUIRE_EQUAL(feature_names.size(), 1u);
        BOOST_CHECK_EQUAL(feature_names[0], "planets");
    }

    // Open attribute without re-opening the dataset.
    {
        std::shared_ptr<fl::Attribute> attribute(dataset->open_attribute(
            "planets/gravity"));
        BOOST_REQUIRE(attribute);
        std::shared_ptr<fl::ConstantAttribute<int32_t>>
            constant_attribute(
                std::dynamic_pointer_cast<fl::ConstantAttribute<int32_t>>(
                    attribute));
        BOOST_REQUIRE(constant_attribute);
    }

    // Read attribute without re-opening the dataset.
    {
        std::shared_ptr<fl::Attribute> attribute(dataset->read_attribute(
            "planets/gravity"));
        BOOST_REQUIRE(attribute);
        std::shared_ptr<fl::ConstantAttribute<int32_t>>
            constant_attribute(
                std::dynamic_pointer_cast<fl::ConstantAttribute<int32_t>>(
                    attribute));
        BOOST_REQUIRE(constant_attribute);
        BOOST_CHECK_EQUAL(constant_attribute->values().value(), 5);
    }

    // Read attribute after the dataset was closed.
    dataset = std::make_shared<fl::FernDataset>(
        "dataset_test_write.gnr", fl::OpenMode::READ);
    {
        std::shared_ptr<fl::Attribute> attribute(dataset->read_attribute(
            "planets/gravity"));
        BOOST_REQUIRE(attribute);
        std::shared_ptr<fl::ConstantAttribute<int32_t>>
            constant_attribute(
                std::dynamic_pointer_cast<fl::ConstantAttribute<int32_t>>(
                    attribute));
        BOOST_REQUIRE(constant_attribute);
        BOOST_CHECK_EQUAL(constant_attribute->values().value(), 5);
    }

    // Update the value.
    dataset = std::make_shared<fl::FernDataset>(
        "dataset_test_write.gnr", fl::OpenMode::UPDATE);
    {
        fl::ConstantAttribute<int32_t> constant(6);
        dataset->write_attribute(constant, "planets/gravity");
        std::shared_ptr<fl::Attribute> attribute(dataset->read_attribute(
            "planets/gravity"));
        BOOST_REQUIRE(attribute);
        std::shared_ptr<fl::ConstantAttribute<int32_t>>
            constant_attribute(
                std::dynamic_pointer_cast<fl::ConstantAttribute<int32_t>>(
                    attribute));
        BOOST_REQUIRE(constant_attribute);
        BOOST_CHECK_EQUAL(constant_attribute->values().value(), 6);
    }
}


BOOST_AUTO_TEST_CASE(errors)
{
    {
        fl::FernDataset dataset("constant-1.h5", fl::OpenMode::READ);

        try {
            dataset.read_feature("blahdiblah");
            BOOST_CHECK(false);
        }
        catch(fern::IOError const& exception) {
            BOOST_CHECK_EQUAL(exception.message(),
                "I/O error handling constant-1.h5: "
                "Does not contain feature: blahdiblah");
        }

        try {
            dataset.read_attribute("blahdiblah");
            BOOST_CHECK(false);
        }
        catch(fern::IOError const& exception) {
            BOOST_CHECK_EQUAL(exception.message(),
                "I/O error handling constant-1.h5: "
                "Does not contain attribute: blahdiblah");
        }
    }
}

BOOST_AUTO_TEST_SUITE_END()
