#include "Ranally/Language/OptimizeVisitorTest.h"
#include <boost/shared_ptr.hpp>
#include <boost/test/test_tools.hpp>
#include <boost/test/unit_test_suite.hpp>
#include "Ranally/Util/String.h"



boost::unit_test::test_suite* OptimizeVisitorTest::suite()
{
  boost::unit_test::test_suite* suite = BOOST_TEST_SUITE(__FILE__);
  boost::shared_ptr<OptimizeVisitorTest> instance(
    new OptimizeVisitorTest());
  suite->add(BOOST_CLASS_TEST_CASE(
    &OptimizeVisitorTest::testRemoveTemporaryIdentifier, instance));

  return suite;
}



OptimizeVisitorTest::OptimizeVisitorTest()

  : _interpreter(),
    _scriptVisitor(),
    _optimizeVisitor()

{
}



void OptimizeVisitorTest::testRemoveTemporaryIdentifier()
{
  namespace rl = ranally::language;

  boost::shared_ptr<rl::ScriptVertex> tree;
  UnicodeString script;

  // Make sure that temporary identifiers which are only used as input to
  // one expression, are removed.
  {
    // This script should be rewritten in the tree as d = f(5).
    script = UnicodeString(
      "a = 5\n"
      "d = f(a)\n");
    tree = _interpreter.parseString(script);
    _interpreter.annotate(tree);
    tree->Accept(_optimizeVisitor);
    tree->Accept(_scriptVisitor);
    std::cout << ranally::util::encodeInUTF8(_scriptVisitor.script()) << std::endl;
    BOOST_CHECK(_scriptVisitor.script() == UnicodeString("d = f(5)"));
  }

  // Make sure that temporary identifiers which are only used as input to
  // more than one expression, are removed.
  {
    // This script should not be rewritten.
    script = UnicodeString(
      "a = 5\n"
      "d = f(a)\n"
      "e = g(a)\n");
    tree = _interpreter.parseString(script);
    _interpreter.annotate(tree);
    tree->Accept(_optimizeVisitor);
    tree->Accept(_scriptVisitor);
    BOOST_CHECK(_scriptVisitor.script() == script);
  }
}

