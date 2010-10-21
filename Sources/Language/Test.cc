#include <boost/test/included/unit_test.hpp>

#include "AlgebraParserTest.h"
#include "AssignmentVertexTest.h"
#include "ExpressionVertexTest.h"
#include "FunctionVertexTest.h"
#include "IfVertexTest.h"
#include "NameVertexTest.h"
#include "NumberVertexTest.h"
#include "ScriptVisitorTest.h"
#include "StatementVertexTest.h"
#include "StringVertexTest.h"
#include "SyntaxTreeTest.h"
#include "SyntaxVertexTest.h"
#include "XmlParserTest.h"



boost::unit_test::test_suite* init_unit_test_suite(
         int argc,
         char** const argv) {

  struct TestSuite: public boost::unit_test::test_suite
  {
    TestSuite(
         int& /* argc */,
         char** /* argv */)
      : boost::unit_test::test_suite("Master test suite")
    {
    }
  };

  TestSuite* test = new TestSuite(argc, argv);

  test->add(AlgebraParserTest::suite());

  test->add(SyntaxVertexTest::suite());
  test->add(NameVertexTest::suite());
  test->add(NumberVertexTest::suite());
  test->add(StringVertexTest::suite());
  test->add(ExpressionVertexTest::suite());
  test->add(FunctionVertexTest::suite());
  test->add(AssignmentVertexTest::suite());
  test->add(StatementVertexTest::suite());
  test->add(IfVertexTest::suite());
  test->add(SyntaxTreeTest::suite());

  test->add(XmlParserTest::suite());

  test->add(ScriptVisitorTest::suite());

  return test;
}

