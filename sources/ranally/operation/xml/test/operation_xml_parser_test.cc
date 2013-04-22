#define BOOST_TEST_MODULE ranally operation_xml
#include <boost/test/unit_test.hpp>
#include "ranally/operation/core/parameter.h"
#include "ranally/operation/core/result.h"
#include "ranally/operation/xml/operation_xml_parser.h"


BOOST_AUTO_TEST_SUITE(operation_xml_parser)

BOOST_AUTO_TEST_CASE(parse)
{
    ranally::OperationXmlParser xml_parser;
    ranally::String xml;
    ranally::OperationsPtr operations;
    std::vector<ranally::Parameter> parameters;
    std::vector<ranally::Result> results;
    ranally::DataTypes data_types;
    ranally::ValueTypes value_types;

    {
        // Empty xml.
        xml =
            "<?xml version=\"1.0\"?>"
            "<Operations/>";
        operations = xml_parser.parse(xml);
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

        operations = xml_parser.parse(xml);
        BOOST_CHECK_EQUAL(operations->size(), 1u);
        BOOST_REQUIRE(operations->has_operation("print"));

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
        data_types = parameter.data_types();
        BOOST_CHECK(data_types == ranally::DataTypes::ALL);

        value_types = parameter.value_types();
        BOOST_CHECK(value_types == ranally::ValueTypes::ALL);

        results = operation->results();
        BOOST_CHECK(results.empty());
    }
}

BOOST_AUTO_TEST_SUITE_END()
