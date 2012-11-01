#include "Ranally/Language/AnnotateVisitorTest.h"
#include <boost/test/test_tools.hpp>
#include <boost/test/unit_test_suite.hpp>
#include "Ranally/Operation/Operation-xml.h"
#include "Ranally/Operation/Parameter.h"
#include "Ranally/Operation/Result.h"
#include "Ranally/Operation/OperationXmlParser.h"
#include "Ranally/Language/OperationVertex.h"
#include "Ranally/Language/ScriptVertex.h"


boost::unit_test::test_suite* AnnotateVisitorTest::suite()
{
    boost::unit_test::test_suite* suite = BOOST_TEST_SUITE(__FILE__);
    boost::shared_ptr<AnnotateVisitorTest> instance(
        new AnnotateVisitorTest());
    suite->add(BOOST_CLASS_TEST_CASE(
        &AnnotateVisitorTest::testVisitEmptyScript, instance));
    suite->add(BOOST_CLASS_TEST_CASE(
        &AnnotateVisitorTest::testVisitNumber, instance));
    suite->add(BOOST_CLASS_TEST_CASE(
        &AnnotateVisitorTest::testVisitOperation, instance));

    return suite;
}


AnnotateVisitorTest::AnnotateVisitorTest()

    : _algebraParser(),
      _xmlParser(),
      _visitor(ranally::OperationXmlParser().parse(
        ranally::operationsXml))

{
}


void AnnotateVisitorTest::testVisitEmptyScript()
{
    // Parse empty script.
    // Ast before and after should be the same.
    std::shared_ptr<ranally::ScriptVertex> tree1, tree2;

    tree1 = _xmlParser.parse(_algebraParser.parseString(ranally::String("")));
    assert(tree1);

    // // Create copy of this empty tree.
    // ranally::CopyVisitor copyVisitor;
    // tree1->Accept(copyVisitor);
    // tree2 = copyVisitor.scriptVertex();

    tree1->Accept(_visitor);

    // // Both trees should be equal.
    // BOOST_CHECK(*tree1 == *tree2);

    BOOST_CHECK_EQUAL(tree1->sourceName(), ranally::String("<string>"));
    BOOST_CHECK_EQUAL(tree1->line(), 0);
    BOOST_CHECK_EQUAL(tree1->col(), 0);
    BOOST_CHECK(tree1->statements().empty());
}


void AnnotateVisitorTest::testVisitNumber()
{
    std::shared_ptr<ranally::ScriptVertex> tree =
        _xmlParser.parse(_algebraParser.parseString(ranally::String("5")));
    tree->Accept(_visitor);

    BOOST_CHECK_EQUAL(tree->sourceName(), ranally::String("<string>"));
    BOOST_CHECK_EQUAL(tree->line(), 0);
    BOOST_CHECK_EQUAL(tree->col(), 0);
    BOOST_CHECK_EQUAL(tree->statements().size(), 1u);
}


void AnnotateVisitorTest::testVisitOperation()
{
    {
        std::shared_ptr<ranally::ScriptVertex> tree =
            _xmlParser.parse(_algebraParser.parseString(
                ranally::String("abs(a)")));
        tree->Accept(_visitor);

        BOOST_CHECK_EQUAL(tree->sourceName(), ranally::String("<string>"));
        BOOST_CHECK_EQUAL(tree->line(), 0);
        BOOST_CHECK_EQUAL(tree->col(), 0);

        BOOST_REQUIRE_EQUAL(tree->statements().size(), 1u);
        std::shared_ptr<ranally::StatementVertex> const& statement(
            tree->statements()[0]);
        BOOST_REQUIRE(statement);
        ranally::OperationVertex const* functionVertex(
            dynamic_cast<ranally::OperationVertex*>(statement.get()));
        BOOST_REQUIRE(functionVertex);

        ranally::OperationPtr const& operation(functionVertex->operation());
        BOOST_REQUIRE(operation);

        BOOST_CHECK_EQUAL(operation->parameters().size(), 1u);
        std::vector<ranally::Parameter> const& parameters(
            operation->parameters());
        ranally::Parameter const& parameter(parameters[0]);
        BOOST_CHECK_EQUAL(parameter.dataTypes(),
            ranally::DataTypes(ranally::DT_VALUE | ranally::DT_RASTER));
        BOOST_CHECK_EQUAL(parameter.valueTypes(),
            ranally::ValueTypes(ranally::ValueType(ranally::VT_NUMBER)));

        BOOST_CHECK_EQUAL(operation->results().size(), 1u);
        std::vector<ranally::Result> const& results(operation->results());
        ranally::Result const& result(results[0]);
        BOOST_CHECK_EQUAL(result.dataType(), ranally::DT_DEPENDS_ON_INPUT);
        BOOST_CHECK_EQUAL(result.valueType(), ranally::VT_DEPENDS_ON_INPUT);
    }
}
