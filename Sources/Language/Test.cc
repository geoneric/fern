#include <boost/test/included/unit_test.hpp>

#include "AlgebraParserTest.h"
#include "AssignmentVertexTest.h"
#include "DotVisitorTest.h"
#include "ExpressionVertexTest.h"
#include "FunctionVertexTest.h"
#include "IfVertexTest.h"
#include "NameVertexTest.h"
#include "NumberVertexTest.h"
#include "OperatorVertexTest.h"
#include "ScriptVertexTest.h"
#include "ScriptVisitorTest.h"
#include "StatementVertexTest.h"
#include "StringVertexTest.h"
#include "SyntaxVertexTest.h"
#include "WhileVertexTest.h"
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
  test->add(OperatorVertexTest::suite());
  test->add(AssignmentVertexTest::suite());
  test->add(StatementVertexTest::suite());
  test->add(IfVertexTest::suite());
  test->add(WhileVertexTest::suite());
  test->add(ScriptVertexTest::suite());

  test->add(XmlParserTest::suite());

  test->add(DotVisitorTest::suite());
  test->add(ScriptVisitorTest::suite());

  return test;
}

