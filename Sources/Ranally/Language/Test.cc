#include <boost/test/included/unit_test.hpp>
#include "Ranally/Language/AlgebraParserTest.h"
#include "Ranally/Language/AnnotateVisitorTest.h"
#include "Ranally/Language/AssignmentVertexTest.h"
#include "Ranally/Language/DotVisitorTest.h"
#include "Ranally/Language/ExecuteVisitorTest.h"
#include "Ranally/Language/ExpressionVertexTest.h"
#include "Ranally/Language/FunctionVertexTest.h"
#include "Ranally/Language/IdentifyVisitorTest.h"
#include "Ranally/Language/IfVertexTest.h"
#include "Ranally/Language/NameVertexTest.h"
#include "Ranally/Language/NumberVertexTest.h"
#include "Ranally/Language/OperatorVertexTest.h"
#include "Ranally/Language/OptimizeVisitorTest.h"
#include "Ranally/Language/ScriptVertexTest.h"
#include "Ranally/Language/ScriptVisitorTest.h"
#include "Ranally/Language/StatementVertexTest.h"
#include "Ranally/Language/StringVertexTest.h"
#include "Ranally/Language/SymbolTableTest.h"
#include "Ranally/Language/SyntaxVertexTest.h"
#include "Ranally/Language/ThreadVisitorTest.h"
#include "Ranally/Language/ValidateVisitorTest.h"
#include "Ranally/Language/VisitorTest.h"
#include "Ranally/Language/WhileVertexTest.h"
#include "Ranally/Language/XmlParserTest.h"


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

    test->add(SymbolTableTest::suite());

    test->add(VisitorTest::suite());
    test->add(ThreadVisitorTest::suite());
    test->add(AnnotateVisitorTest::suite());
    test->add(IdentifyVisitorTest::suite());
    test->add(ValidateVisitorTest::suite());
    // test->add(OptimizeVisitorTest::suite());
    test->add(ExecuteVisitorTest::suite());

    test->add(DotVisitorTest::suite());
    test->add(ScriptVisitorTest::suite());

    return test;
}
