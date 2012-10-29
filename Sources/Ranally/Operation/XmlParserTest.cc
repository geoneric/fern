#include "XmlParserTest.h"
#include <boost/shared_ptr.hpp>
#include <boost/test/test_tools.hpp>
#include <boost/test/unit_test_suite.hpp>
#include "Ranally/Operation/Parameter.h"
#include "Ranally/Operation/Result.h"
#include "Ranally/Operation/XmlParser.h"


boost::unit_test::test_suite* XmlParserTest::suite()
{
    boost::unit_test::test_suite* suite = BOOST_TEST_SUITE(__FILE__);
    boost::shared_ptr<XmlParserTest> instance(
        new XmlParserTest());
    suite->add(BOOST_CLASS_TEST_CASE(
        &XmlParserTest::testParse, instance));

    return suite;
}


XmlParserTest::XmlParserTest()
{
}


void XmlParserTest::testParse()
{
    ranally::operation::XmlParser xmlParser;
    ranally::String xml;
    ranally::operation::OperationsPtr operations;
    std::vector<ranally::operation::Parameter> parameters;
    std::vector<ranally::operation::Result> results;
    ranally::operation::DataTypes dataTypes;
    ranally::operation::ValueTypes valueTypes;

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

        ranally::operation::OperationPtr const& operation(
            operations->operation("print"));
        BOOST_CHECK(operation->name() == "print");
        BOOST_CHECK(operation->description() ==
            "Print the argument value to the standard output stream.");

        parameters = operation->parameters();
        BOOST_CHECK_EQUAL(parameters.size(), 1u);
        ranally::operation::Parameter parameter = parameters[0];
        BOOST_CHECK(parameter.name() == "value");
        BOOST_CHECK(parameter.description() == "Value to print.");
        dataTypes = parameter.dataTypes();
        BOOST_CHECK(dataTypes == ranally::operation::DT_ALL);

        valueTypes = parameter.valueTypes();
        BOOST_CHECK(valueTypes == ranally::operation::VT_ALL);

        results = operation->results();
        BOOST_CHECK(results.empty());
    }
}
