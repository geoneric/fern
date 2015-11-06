// -----------------------------------------------------------------------------
// Fern © Geoneric
//
// This file is part of Geoneric Fern which is available under the terms of
// the GNU General Public License (GPL), version 2. If you do not want to
// be bound by the terms of the GPL, you may purchase a proprietary license
// from Geoneric (http://www.geoneric.eu/contact).
// -----------------------------------------------------------------------------
#define BOOST_TEST_MODULE fern operation xml operation_xml_parser
#include <boost/test/unit_test.hpp>
#include "fern/language/operation/core/parameter.h"
#include "fern/language/operation/core/result.h"
#include "fern/language/operation/xml/operation_xml_parser.h"


namespace fl = fern::language;


BOOST_AUTO_TEST_CASE(parse)
{
    fl::OperationXmlParser xml_parser;
    std::string xml;
    fl::OperationsPtr operations;
    std::vector<fl::Parameter> parameters;
    std::vector<fl::Result> results;
    fern::DataTypes data_types;
    fern::ValueTypes value_types;

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

        // TODO?
        // operations = xml_parser.parse(xml);
        // BOOST_CHECK_EQUAL(operations->size(), 1u);
        // BOOST_REQUIRE(operations->has_operation("print"));

        // fern::OperationPtr const& operation(
        //     operations->operation("print"));
        // BOOST_CHECK(operation->name() == "print");
        // BOOST_CHECK(operation->description() ==
        //     "Print the argument value to the standard output stream.");

        // parameters = operation->parameters();
        // BOOST_CHECK_EQUAL(parameters.size(), 1u);
        // fern::Parameter parameter = parameters[0];
        // BOOST_CHECK(parameter.name() == "value");
        // BOOST_CHECK(parameter.description() == "Value to print.");
        // assert(parameter.expression_types().size() == 1u);
        // data_types = parameter.expression_types()[0].data_type();
        // BOOST_CHECK(data_types == fern::DataTypes::ALL);

        // value_types = parameter.expression_types()[0].value_type();
        // BOOST_CHECK(value_types == fern::ValueTypes::ALL);

        // results = operation->results();
        // BOOST_CHECK(results.empty());
    }
}
