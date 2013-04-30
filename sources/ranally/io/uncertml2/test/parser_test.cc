#define BOOST_TEST_MODULE ranally operation_xml
#include <boost/test/unit_test.hpp>
#include "ranally/io/uncertml2/uncertml2_parser.h"


BOOST_AUTO_TEST_SUITE(io_uncertml2_parser)

BOOST_AUTO_TEST_CASE(parse)
{
    ranally::UncertML2Parser parser;
    ranally::String xml;
    std::shared_ptr<ranally::Uncertainty> uncertainty;

    // std::vector<ranally::Parameter> parameters;
    // std::vector<ranally::Result> results;
    // ranally::DataTypes data_types;
    // ranally::ValueTypes value_types;

    {
        // Empty xml.
        // TODO Figure out how to get at the diagnostics. Current error message
        //      is bull.
        xml =
            "<?xml version=\"1.0\"?>"
            "<un:NormalDistribution xmlns:un=\"http://www.uncertml.org/2.0\">"
              "<un:mean>3.14</un:mean>"
              "<un:variance>3.14</un:variance>"
            "</un:NormalDistribution>"
            ;
        try {
        uncertainty = parser.parse(xml);
        } catch(std::exception const& exception) {
            std::cout << exception.what() << std::endl;
        }
        BOOST_REQUIRE(false);
        BOOST_REQUIRE(uncertainty);
    }

    // {
    //     xml =
    //         "<?xml version=\"1.0\"?>"
    //         "<Operations>"
    //           "<Operation>"
    //             "<Name>print</Name>"
    //             "<Description>Print the argument value to the standard output stream.</Description>"
    //             "<Parameters>"
    //               "<Parameter>"
    //                 "<Name>value</Name>"
    //                 "<Description>Value to print.</Description>"
    //                 "<DataTypes>"
    //                   "<DataType>All</DataType>"
    //                 "</DataTypes>"
    //                 "<ValueTypes>"
    //                   "<ValueType>All</ValueType>"
    //                 "</ValueTypes>"
    //               "</Parameter>"
    //             "</Parameters>"
    //             "<Results/>"
    //           "</Operation>"
    //         "</Operations>";

    //     operations = parser.parse(xml);
    //     BOOST_CHECK_EQUAL(operations->size(), 1u);
    //     BOOST_REQUIRE(operations->has_operation("print"));

    //     ranally::OperationPtr const& operation(
    //         operations->operation("print"));
    //     BOOST_CHECK(operation->name() == "print");
    //     BOOST_CHECK(operation->description() ==
    //         "Print the argument value to the standard output stream.");

    //     parameters = operation->parameters();
    //     BOOST_CHECK_EQUAL(parameters.size(), 1u);
    //     ranally::Parameter parameter = parameters[0];
    //     BOOST_CHECK(parameter.name() == "value");
    //     BOOST_CHECK(parameter.description() == "Value to print.");
    //     data_types = parameter.data_types();
    //     BOOST_CHECK(data_types == ranally::DataTypes::ALL);

    //     value_types = parameter.value_types();
    //     BOOST_CHECK(value_types == ranally::ValueTypes::ALL);

    //     results = operation->results();
    //     BOOST_CHECK(results.empty());
    // }
}

BOOST_AUTO_TEST_SUITE_END()
