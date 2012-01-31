#include "Ranally/Language/ThreadVisitorTest.h"
#include <iostream>
#include <boost/shared_ptr.hpp>
#include <boost/test/test_tools.hpp>
#include <boost/test/unit_test_suite.hpp>
#include "Ranally/Util/String.h"
#include "Ranally/Language/AssignmentVertex.h"
#include "Ranally/Language/FunctionVertex.h"
#include "Ranally/Language/IfVertex.h"
#include "Ranally/Language/OperatorVertex.h"
#include "Ranally/Language/ScriptVertex.h"



boost::unit_test::test_suite* ThreadVisitorTest::suite()
{
  boost::unit_test::test_suite* suite = BOOST_TEST_SUITE(__FILE__);
  boost::shared_ptr<ThreadVisitorTest> instance(
    new ThreadVisitorTest());
  suite->add(BOOST_CLASS_TEST_CASE(
    &ThreadVisitorTest::testVisitEmptyScript, instance));
  suite->add(BOOST_CLASS_TEST_CASE(
    &ThreadVisitorTest::testVisitName, instance));
  suite->add(BOOST_CLASS_TEST_CASE(
    &ThreadVisitorTest::testVisitAssignment, instance));
  suite->add(BOOST_CLASS_TEST_CASE(
    &ThreadVisitorTest::testVisitString, instance));
  suite->add(BOOST_CLASS_TEST_CASE(
    &ThreadVisitorTest::testVisitNumber, instance));
  suite->add(BOOST_CLASS_TEST_CASE(
    &ThreadVisitorTest::testVisitFunction, instance));
  suite->add(BOOST_CLASS_TEST_CASE(
    &ThreadVisitorTest::testVisitOperator, instance));
  suite->add(BOOST_CLASS_TEST_CASE(
    &ThreadVisitorTest::testVisitMultipleStatements, instance));
  suite->add(BOOST_CLASS_TEST_CASE(
    &ThreadVisitorTest::testVisitNestedExpression, instance));
  suite->add(BOOST_CLASS_TEST_CASE(
    &ThreadVisitorTest::testVisitIf, instance));
  suite->add(BOOST_CLASS_TEST_CASE(
    &ThreadVisitorTest::testVisitWhile, instance));

  return suite;
}



ThreadVisitorTest::ThreadVisitorTest()
{
}



void ThreadVisitorTest::testVisitEmptyScript()
{
  boost::shared_ptr<ranally::language::ScriptVertex> tree;

  tree = _xmlParser.parse(_algebraParser.parseString(UnicodeString("")));
  tree->Accept(_visitor);
  BOOST_CHECK_EQUAL(tree->successor(), &(*tree));
}



void ThreadVisitorTest::testVisitName()
{
  boost::shared_ptr<ranally::language::ScriptVertex> tree;

  tree = _xmlParser.parse(_algebraParser.parseString(UnicodeString("a")));
  tree->Accept(_visitor);

  ranally::language::SyntaxVertex const* vertexA = &(*tree->statements()[0]);

  BOOST_CHECK_EQUAL(tree->successor(), vertexA);
  BOOST_CHECK_EQUAL(vertexA->successor(), &(*tree));
}



void ThreadVisitorTest::testVisitAssignment()
{
  boost::shared_ptr<ranally::language::ScriptVertex> tree;

  tree = _xmlParser.parse(_algebraParser.parseString(UnicodeString("a = b")));
  tree->Accept(_visitor);

  ranally::language::AssignmentVertex const* assignment =
    dynamic_cast<ranally::language::AssignmentVertex const*>(
      &(*tree->statements()[0]));
  ranally::language::SyntaxVertex const* vertexA = &(*assignment->target());
  ranally::language::SyntaxVertex const* vertexB =
    &(*assignment->expression());

  BOOST_CHECK_EQUAL(tree->successor(), vertexB);
  BOOST_CHECK_EQUAL(vertexB->successor(), vertexA);
  BOOST_CHECK_EQUAL(vertexA->successor(), assignment);
  BOOST_CHECK_EQUAL(assignment->successor(), &(*tree));
}



void ThreadVisitorTest::testVisitString()
{
  boost::shared_ptr<ranally::language::ScriptVertex> tree;

  tree = _xmlParser.parse(_algebraParser.parseString(UnicodeString(
    "\"five\"")));
  tree->Accept(_visitor);

  ranally::language::SyntaxVertex const* stringVertex = &(*tree->statements()[0]);

  BOOST_CHECK_EQUAL(tree->successor(), stringVertex);
  BOOST_CHECK_EQUAL(stringVertex->successor(), &(*tree));
}



void ThreadVisitorTest::testVisitNumber()
{
  boost::shared_ptr<ranally::language::ScriptVertex> tree;

  tree = _xmlParser.parse(_algebraParser.parseString(UnicodeString("5")));
  tree->Accept(_visitor);

  ranally::language::SyntaxVertex const* numberVertex = &(*tree->statements()[0]);

  BOOST_CHECK_EQUAL(tree->successor(), numberVertex);
  BOOST_CHECK_EQUAL(numberVertex->successor(), &(*tree));
}



void ThreadVisitorTest::testVisitFunction()
{
  boost::shared_ptr<ranally::language::ScriptVertex> tree;

  {
    tree = _xmlParser.parse(_algebraParser.parseString(UnicodeString("f()")));
    tree->Accept(_visitor);

    ranally::language::SyntaxVertex const* functionVertex =
      &(*tree->statements()[0]);

    BOOST_CHECK_EQUAL(tree->successor(), functionVertex);
    BOOST_CHECK_EQUAL(functionVertex->successor(), &(*tree));
  }

  {
    tree = _xmlParser.parse(_algebraParser.parseString(UnicodeString(
      "f(1, \"2\", three, four())")));
    tree->Accept(_visitor);

    ranally::language::FunctionVertex const* functionVertex =
      dynamic_cast<ranally::language::FunctionVertex const*>(
        &(*tree->statements()[0]));
    ranally::language::SyntaxVertex const* vertex1 =
      &(*functionVertex->expressions()[0]);
    ranally::language::SyntaxVertex const* vertex2 =
      &(*functionVertex->expressions()[1]);
    ranally::language::SyntaxVertex const* vertex3 =
      &(*functionVertex->expressions()[2]);
    ranally::language::SyntaxVertex const* vertex4 =
      &(*functionVertex->expressions()[3]);

    BOOST_CHECK_EQUAL(tree->successor(), vertex1);
    BOOST_CHECK_EQUAL(vertex1->successor(), vertex2);
    BOOST_CHECK_EQUAL(vertex2->successor(), vertex3);
    BOOST_CHECK_EQUAL(vertex3->successor(), vertex4);
    BOOST_CHECK_EQUAL(vertex4->successor(), functionVertex);
    BOOST_CHECK_EQUAL(functionVertex->successor(), &(*tree));
  }
}



void ThreadVisitorTest::testVisitOperator()
{
  boost::shared_ptr<ranally::language::ScriptVertex> tree;

  {
    tree = _xmlParser.parse(_algebraParser.parseString(UnicodeString("-a")));
    tree->Accept(_visitor);

    ranally::language::OperatorVertex const* operatorVertex =
      dynamic_cast<ranally::language::OperatorVertex const*>(
        &(*tree->statements()[0]));
    ranally::language::SyntaxVertex const* vertex1 =
      &(*operatorVertex->expressions()[0]);

    BOOST_CHECK_EQUAL(tree->successor(), vertex1);
    BOOST_CHECK_EQUAL(vertex1->successor(), operatorVertex);
    BOOST_CHECK_EQUAL(operatorVertex->successor(), &(*tree));
  }

  {
    tree = _xmlParser.parse(_algebraParser.parseString(UnicodeString("a + b")));
    tree->Accept(_visitor);

    ranally::language::OperatorVertex const* operatorVertex =
      dynamic_cast<ranally::language::OperatorVertex const*>(
        &(*tree->statements()[0]));
    ranally::language::SyntaxVertex const* vertex1 =
      &(*operatorVertex->expressions()[0]);
    ranally::language::SyntaxVertex const* vertex2 =
      &(*operatorVertex->expressions()[1]);

    BOOST_CHECK_EQUAL(tree->successor(), vertex1);
    BOOST_CHECK_EQUAL(vertex1->successor(), vertex2);
    BOOST_CHECK_EQUAL(vertex2->successor(), operatorVertex);
    BOOST_CHECK_EQUAL(operatorVertex->successor(), &(*tree));
  }

  {
    tree = _xmlParser.parse(_algebraParser.parseString(UnicodeString(
      "-(a + b)")));
    tree->Accept(_visitor);

    ranally::language::OperatorVertex const* operator1Vertex =
      dynamic_cast<ranally::language::OperatorVertex const*>(
        &(*tree->statements()[0]));
    ranally::language::OperatorVertex const* operator2Vertex =
      dynamic_cast<ranally::language::OperatorVertex const*>(
        &(*operator1Vertex->expressions()[0]));
    ranally::language::SyntaxVertex const* vertex1 =
      &(*operator2Vertex->expressions()[0]);
    ranally::language::SyntaxVertex const* vertex2 =
      &(*operator2Vertex->expressions()[1]);

    BOOST_CHECK_EQUAL(tree->successor(), vertex1);
    BOOST_CHECK_EQUAL(vertex1->successor(), vertex2);
    BOOST_CHECK_EQUAL(vertex2->successor(), operator2Vertex);
    BOOST_CHECK_EQUAL(operator2Vertex->successor(), operator1Vertex);
    BOOST_CHECK_EQUAL(operator1Vertex->successor(), &(*tree));
  }
}



void ThreadVisitorTest::testVisitMultipleStatements()
{
  boost::shared_ptr<ranally::language::ScriptVertex> tree;

  tree = _xmlParser.parse(_algebraParser.parseString(UnicodeString("a;b;c")));
  tree->Accept(_visitor);

  ranally::language::SyntaxVertex const* vertexA = &(*tree->statements()[0]);
  ranally::language::SyntaxVertex const* vertexB = &(*tree->statements()[1]);
  ranally::language::SyntaxVertex const* vertexC = &(*tree->statements()[2]);

  BOOST_CHECK_EQUAL(tree->successor(), vertexA);
  BOOST_CHECK_EQUAL(vertexA->successor(), vertexB);
  BOOST_CHECK_EQUAL(vertexB->successor(), vertexC);
  BOOST_CHECK_EQUAL(vertexC->successor(), &(*tree));
}



void ThreadVisitorTest::testVisitNestedExpression()
{
  boost::shared_ptr<ranally::language::ScriptVertex> tree;

  tree = _xmlParser.parse(_algebraParser.parseString(UnicodeString(
    "a = b + c")));
  tree->Accept(_visitor);

  ranally::language::AssignmentVertex const* assignment =
    dynamic_cast<ranally::language::AssignmentVertex const*>(
      &(*tree->statements()[0]));
  ranally::language::SyntaxVertex const* vertexA = &(*assignment->target());
  ranally::language::OperatorVertex const* addition =
    dynamic_cast<ranally::language::OperatorVertex const*>(
      &(*assignment->expression()));
  ranally::language::SyntaxVertex const* vertexB =
    &(*addition->expressions()[0]);
  ranally::language::SyntaxVertex const* vertexC =
    &(*addition->expressions()[1]);

  BOOST_CHECK_EQUAL(tree->successor(), vertexB);
  BOOST_CHECK_EQUAL(vertexB->successor(), vertexC);
  BOOST_CHECK_EQUAL(vertexC->successor(), addition);
  BOOST_CHECK_EQUAL(addition->successor(), vertexA);
  BOOST_CHECK_EQUAL(vertexA->successor(), assignment);
  BOOST_CHECK_EQUAL(assignment->successor(), &(*tree));
}



void ThreadVisitorTest::testVisitIf()
{
  boost::shared_ptr<ranally::language::ScriptVertex> tree;

  {
    tree = _xmlParser.parse(_algebraParser.parseString(UnicodeString(
      "if a:\n"
      "  b\n"
      "  c")));
    tree->Accept(_visitor);

    ranally::language::IfVertex const* ifVertex =
      dynamic_cast<ranally::language::IfVertex const*>(
        &(*tree->statements()[0]));
    ranally::language::SyntaxVertex const* vertexA = &(*ifVertex->condition());
    ranally::language::SyntaxVertex const* vertexB =
      &(*ifVertex->trueStatements()[0]);
    ranally::language::SyntaxVertex const* vertexC =
      &(*ifVertex->trueStatements()[1]);

    BOOST_CHECK_EQUAL(tree->successor(), vertexA);
    BOOST_CHECK_EQUAL(vertexA->successor(), ifVertex);
    BOOST_CHECK_EQUAL(ifVertex->successor(0), vertexB);
    BOOST_CHECK_EQUAL(vertexB->successor(), vertexC);
    BOOST_CHECK_EQUAL(vertexC->successor(), ifVertex);
    BOOST_CHECK_EQUAL(ifVertex->successor(1), &(*tree));
  }

  {
    tree = _xmlParser.parse(_algebraParser.parseString(UnicodeString(
      "if a:\n"
      "  b\n"
      "  c\n"
      "elif d:\n"
      "  e\n"
      "  f\n")));
    tree->Accept(_visitor);

    // True block first if.
    ranally::language::IfVertex const* if1Vertex =
      dynamic_cast<ranally::language::IfVertex const*>(
        &(*tree->statements()[0]));
    ranally::language::SyntaxVertex const* vertexA = &(*if1Vertex->condition());
    ranally::language::SyntaxVertex const* vertexB =
      &(*if1Vertex->trueStatements()[0]);
    ranally::language::SyntaxVertex const* vertexC =
      &(*if1Vertex->trueStatements()[1]);

    // True block second if.
    ranally::language::IfVertex const* if2Vertex =
      dynamic_cast<ranally::language::IfVertex const*>(
        &(*if1Vertex->falseStatements()[0]));
    ranally::language::SyntaxVertex const* vertexD =
      &(*if2Vertex->condition());
    ranally::language::SyntaxVertex const* vertexE =
      &(*if2Vertex->trueStatements()[0]);
    ranally::language::SyntaxVertex const* vertexF =
      &(*if2Vertex->trueStatements()[1]);

    // True block first if.
    BOOST_CHECK_EQUAL(tree->successor(), vertexA);
    BOOST_CHECK_EQUAL(vertexA->successor(), if1Vertex);
    BOOST_CHECK_EQUAL(if1Vertex->successor(0), vertexB);
    BOOST_CHECK_EQUAL(vertexB->successor(), vertexC);
    BOOST_CHECK_EQUAL(vertexC->successor(), if1Vertex);

    // True block second if.
    BOOST_CHECK_EQUAL(if1Vertex->successor(1), vertexD);
    BOOST_CHECK_EQUAL(vertexD->successor(), if2Vertex);
    BOOST_CHECK_EQUAL(if2Vertex->successor(0), vertexE);
    BOOST_CHECK_EQUAL(vertexE->successor(), vertexF);
    BOOST_CHECK_EQUAL(vertexF->successor(), if2Vertex);

    BOOST_CHECK_EQUAL(if2Vertex->successor(1), if1Vertex);

    BOOST_CHECK_EQUAL(if1Vertex->successor(2), &(*tree));
  }

  {
    tree = _xmlParser.parse(_algebraParser.parseString(UnicodeString(
      "if a:\n"
      "  b\n"
      "  c\n"
      "elif d:\n"
      "  e\n"
      "  f\n"
      "else:\n"
      "  g\n"
      "  h\n")));
    tree->Accept(_visitor);

    // True block first if.
    ranally::language::IfVertex const* if1Vertex =
      dynamic_cast<ranally::language::IfVertex const*>(
        &(*tree->statements()[0]));
    ranally::language::SyntaxVertex const* vertexA = &(*if1Vertex->condition());
    ranally::language::SyntaxVertex const* vertexB =
      &(*if1Vertex->trueStatements()[0]);
    ranally::language::SyntaxVertex const* vertexC =
      &(*if1Vertex->trueStatements()[1]);

    // True block second if.
    ranally::language::IfVertex const* if2Vertex =
      dynamic_cast<ranally::language::IfVertex const*>(
        &(*if1Vertex->falseStatements()[0]));
    ranally::language::SyntaxVertex const* vertexD = &(*if2Vertex->condition());
    ranally::language::SyntaxVertex const* vertexE =
      &(*if2Vertex->trueStatements()[0]);
    ranally::language::SyntaxVertex const* vertexF =
      &(*if2Vertex->trueStatements()[1]);

    // False block second if.
    ranally::language::SyntaxVertex const* vertexG =
      &(*if2Vertex->falseStatements()[0]);
    ranally::language::SyntaxVertex const* vertexH =
      &(*if2Vertex->falseStatements()[1]);

    // True block first if.
    BOOST_CHECK_EQUAL(tree->successor(), vertexA);
    BOOST_CHECK_EQUAL(vertexA->successor(), if1Vertex);
    BOOST_CHECK_EQUAL(if1Vertex->successor(0), vertexB);
    BOOST_CHECK_EQUAL(vertexB->successor(), vertexC);
    BOOST_CHECK_EQUAL(vertexC->successor(), if1Vertex);

    // True block second if.
    BOOST_CHECK_EQUAL(if1Vertex->successor(1), vertexD);
    BOOST_CHECK_EQUAL(vertexD->successor(), if2Vertex);
    BOOST_CHECK_EQUAL(if2Vertex->successor(0), vertexE);
    BOOST_CHECK_EQUAL(vertexE->successor(), vertexF);
    BOOST_CHECK_EQUAL(vertexF->successor(), if2Vertex);

    // False block second if.
    BOOST_CHECK_EQUAL(if2Vertex->successor(1), vertexG);
    BOOST_CHECK_EQUAL(vertexG->successor(), vertexH);
    BOOST_CHECK_EQUAL(vertexH->successor(), if2Vertex);

    BOOST_CHECK_EQUAL(if2Vertex->successor(2), if1Vertex);

    BOOST_CHECK_EQUAL(if1Vertex->successor(2), &(*tree));
  }
}



void ThreadVisitorTest::testVisitWhile()
{
}

