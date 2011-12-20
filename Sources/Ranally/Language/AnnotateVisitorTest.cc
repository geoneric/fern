#include "Ranally/Language/AnnotateVisitorTest.h"
#include <boost/shared_ptr.hpp>
#include <boost/test/test_tools.hpp>
#include <boost/test/unit_test_suite.hpp>
#include "Ranally/Operation/Operation-xml.h"
#include "Ranally/Operation/Parameter.h"
#include "Ranally/Operation/Result.h"
#include "Ranally/Operation/XmlParser.h"
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
    _visitor(ranally::operation::XmlParser().parse(
      ranally::operation::operationsXml))

{
}



void AnnotateVisitorTest::testVisitEmptyScript()
{
  namespace rl = ranally::language;

  // Parse empty script.
  // Ast before and after should be the same.
  boost::shared_ptr<rl::ScriptVertex> tree1, tree2;

  tree1 = _xmlParser.parse(_algebraParser.parseString(UnicodeString("")));
  assert(tree1);

  // // Create copy of this empty tree.
  // rl::CopyVisitor copyVisitor;
  // tree1->Accept(copyVisitor);
  // tree2 = copyVisitor.scriptVertex();

  tree1->Accept(_visitor);

  // // Both trees should be equal.
  // BOOST_CHECK(*tree1 == *tree2);

  BOOST_CHECK(tree1->sourceName() == "<string>");
  BOOST_CHECK_EQUAL(tree1->line(), 0);
  BOOST_CHECK_EQUAL(tree1->col(), 0);
  BOOST_CHECK(tree1->statements().empty());
}



void AnnotateVisitorTest::testVisitNumber()
{
  namespace rl = ranally::language;

  boost::shared_ptr<rl::ScriptVertex> tree =
    _xmlParser.parse(_algebraParser.parseString(UnicodeString("5")));
  tree->Accept(_visitor);

  BOOST_CHECK(tree->sourceName() == "<string>");
  BOOST_CHECK_EQUAL(tree->line(), 0);
  BOOST_CHECK_EQUAL(tree->col(), 0);
  BOOST_CHECK_EQUAL(tree->statements().size(), 1u);
}



void AnnotateVisitorTest::testVisitOperation()
{
  namespace ro = ranally::operation;
  namespace rl = ranally::language;

  {
    boost::shared_ptr<rl::ScriptVertex> tree =
      _xmlParser.parse(_algebraParser.parseString(UnicodeString("abs(a)")));
    tree->Accept(_visitor);

    BOOST_CHECK(tree->sourceName() == "<string>");
    BOOST_CHECK_EQUAL(tree->line(), 0);
    BOOST_CHECK_EQUAL(tree->col(), 0);

    BOOST_REQUIRE_EQUAL(tree->statements().size(), 1u);
    boost::shared_ptr<rl::StatementVertex> const& statement(
      tree->statements()[0]);
    BOOST_REQUIRE(statement);
    rl::OperationVertex const* functionVertex(
      dynamic_cast<rl::OperationVertex*>(statement.get()));
    BOOST_REQUIRE(functionVertex);

    ro::OperationPtr const& operation(functionVertex->operation());
    BOOST_REQUIRE(operation);

    BOOST_CHECK_EQUAL(operation->parameters().size(), 1u);
    std::vector<ro::Parameter> const& parameters(operation->parameters());
    ro::Parameter const& parameter(parameters[0]);
    BOOST_CHECK_EQUAL(parameter.dataTypes(),
      ro::DataTypes(ro::DT_VALUE | ro::DT_RASTER));
    BOOST_CHECK_EQUAL(parameter.valueTypes(),
      ro::ValueTypes(ro::ValueType(ro::VT_NUMBER)));

    BOOST_CHECK_EQUAL(operation->results().size(), 1u);
    std::vector<ro::Result> const& results(operation->results());
    ro::Result const& result(results[0]);
    BOOST_CHECK_EQUAL(result.dataType(), ro::DT_DEPENDS_ON_INPUT);
    BOOST_CHECK_EQUAL(result.valueType(), ro::VT_DEPENDS_ON_INPUT);
  }
}

