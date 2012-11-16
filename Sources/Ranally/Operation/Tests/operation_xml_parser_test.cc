#define BOOST_TEST_MODULE ranally operation
#include <boost/test/included/unit_test.hpp>
#include "Ranally/Operation/Parameter.h"
#include "Ranally/Operation/Result.h"
#include "Ranally/Operation/OperationXmlParser.h"


BOOST_AUTO_TEST_SUITE(operation_xml_parser)

BOOST_AUTO_TEST_CASE(parse)
{
    ranally::OperationXmlParser xmlParser;
    ranally::String xml;
    ranally::OperationsPtr operations;
    std::vector<ranally::Parameter> parameters;
    std::vector<ranally::Result> results;
    ranally::DataTypes dataTypes;
    ranally::ValueTypes valueTypes;

    {
        // Empty xml.
        xml =
            "<?xml version=\"1.0\"?>"
            "<Operations/>";
        operations = xmlParser.parse(xml);
        BOOST_CHECK(operations->empty());
    }

    {
        xml =
            "<?xml version=\"1.0\"?>"
            "<Operations>"
              "<Operation>"
                "<Name>print</Name>"
                "<Description>Print the argument value to the standard output stream.</Description>"
                "<Parameters>"
                  "<Parameter>"
                    "<Name>value</Name>"
                    "<Description>Value to print.</Description>"
                    "<DataTypes>"
                      "<DataType>All</DataType>"
                    "</DataTypes>"
                    "<ValueTypes>"
                      "<ValueType>All</ValueType>"
                    "</ValueTypes>"
                  "</Parameter>"
                "</Parameters>"
                "<Results/>"
              "</Operation>"
            "</Operations>";

        operations = xmlParser.parse(xml);
        BOOST_CHECK_EQUAL(operations->size(), 1u);
        BOOST_REQUIRE(operations->hasOperation("print"));

        ranally::OperationPtr const& operation(
            operations->operation("print"));
        BOOST_CHECK(operation->name() == "print");
        BOOST_CHECK(operation->description() ==
            "Print the argument value to the standard output stream.");

        parameters = operation->parameters();
        BOOST_CHECK_EQUAL(parameters.size(), 1u);
        ranally::Parameter parameter = parameters[0];
        BOOST_CHECK(parameter.name() == "value");
        BOOST_CHECK(parameter.description() == "Value to print.");
        dataTypes = parameter.dataTypes();
        BOOST_CHECK(dataTypes == ranally::DT_ALL);

        valueTypes = parameter.valueTypes();
        BOOST_CHECK(valueTypes == ranally::VT_ALL);

        results = operation->results();
        BOOST_CHECK(results.empty());
    }
}

BOOST_AUTO_TEST_SUITE_END()