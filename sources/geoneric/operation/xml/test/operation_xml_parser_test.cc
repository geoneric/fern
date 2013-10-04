#define BOOST_TEST_MODULE geoneric operation_xml
#include <boost/test/unit_test.hpp>
#include "geoneric/operation/core/parameter.h"
#include "geoneric/operation/core/result.h"
#include "geoneric/operation/xml/operation_xml_parser.h"


BOOST_AUTO_TEST_SUITE(operation_xml_parser)

BOOST_AUTO_TEST_CASE(parse)
{
    geoneric::OperationXmlParser xml_parser;
    geoneric::String xml;
    geoneric::OperationsPtr operations;
    std::vector<geoneric::Parameter> parameters;
    std::vector<geoneric::Result> results;
    geoneric::DataTypes data_types;
    geoneric::ValueTypes value_types;

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

        geoneric::OperationPtr const& operation(
            operations->operation("print"));
        BOOST_CHECK(operation->name() == "print");
        BOOST_CHECK(operation->description() ==
            "Print the argument value to the standard output stream.");

        parameters = operation->parameters();
        BOOST_CHECK_EQUAL(parameters.size(), 1u);
        geoneric::Parameter parameter = parameters[0];
        BOOST_CHECK(parameter.name() == "value");
        BOOST_CHECK(parameter.description() == "Value to print.");
        assert(parameter.result_types().size() == 1u);
        data_types = parameter.result_types()[0].data_type();
        BOOST_CHECK(data_types == geoneric::DataTypes::ALL);

        value_types = parameter.result_types()[0].value_type();
        BOOST_CHECK(value_types == geoneric::ValueTypes::ALL);

        results = operation->results();
        BOOST_CHECK(results.empty());
    }
}

BOOST_AUTO_TEST_SUITE_END()
