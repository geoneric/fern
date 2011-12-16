#include "Ranally/Language/AnnotateVisitorTest.h"
#include <boost/shared_ptr.hpp>
#include <boost/test/test_tools.hpp>
#include <boost/test/unit_test_suite.hpp>
// #include "Ranally/Language/CopyVisitor.h"
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

    // TODO
    // boost::shared_ptr<rl::operation::Requirements> const&
    //   requirements(functionVertex->requirements());
    // BOOST_REQUIRE(requirements);

    // BOOST_CHECK_EQUAL(requirements->arguments().size(), 1u);
    // std::vector<rl::operation::Argument> const& arguments(
    //   requirements->arguments());
    // rl::operation::Argument const& argument(arguments[0]);
    // BOOST_CHECK_EQUAL(argument.dataTypes().count(), 1u);
    // BOOST_CHECK(argument.dataTypes().test(rl::operation::Scalar));
    // BOOST_CHECK_EQUAL(argument.valueTypes().count(), 1u);
    // BOOST_CHECK(argument.valueTypes().test(rl::operation::Number));

    // BOOST_CHECK_EQUAL(requirements->results().size(), 1u);
    // std::vector<rl::operation::Result> const& results(
    //   requirements->results());
    // rl::operation::Result const& result(results[0]);
    // BOOST_CHECK_EQUAL(result.dataTypes().count(), 1u);
    // BOOST_CHECK(result.dataTypes().test(rl::operation::AsArgument));
    // BOOST_CHECK_EQUAL(result.valueTypes().count(), 1u);
    // BOOST_CHECK(result.valueTypes().test(rl::operation::Number));
  }
}

