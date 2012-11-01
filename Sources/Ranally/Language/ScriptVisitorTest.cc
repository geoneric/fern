#include "Ranally/Language/ScriptVisitorTest.h"
#include <iostream>
#include <boost/test/test_tools.hpp>
#include <boost/test/unit_test_suite.hpp>
#include "Ranally/Util/String.h"
#include "Ranally/Language/ScriptVertex.h"


boost::unit_test::test_suite* ScriptVisitorTest::suite()
{
    boost::unit_test::test_suite* suite = BOOST_TEST_SUITE(__FILE__);
    boost::shared_ptr<ScriptVisitorTest> instance(
        new ScriptVisitorTest());
    suite->add(BOOST_CLASS_TEST_CASE(
        &ScriptVisitorTest::testVisitEmptyScript, instance));
    suite->add(BOOST_CLASS_TEST_CASE(
        &ScriptVisitorTest::testVisitName, instance));
    suite->add(BOOST_CLASS_TEST_CASE(
        &ScriptVisitorTest::testVisitAssignment, instance));
    suite->add(BOOST_CLASS_TEST_CASE(
        &ScriptVisitorTest::testVisitString, instance));
    suite->add(BOOST_CLASS_TEST_CASE(
        &ScriptVisitorTest::testVisitNumber, instance));
    suite->add(BOOST_CLASS_TEST_CASE(
        &ScriptVisitorTest::testVisitFunction, instance));
    suite->add(BOOST_CLASS_TEST_CASE(
        &ScriptVisitorTest::testVisitOperator, instance));
    suite->add(BOOST_CLASS_TEST_CASE(
        &ScriptVisitorTest::testVisitMultipleStatements, instance));
    suite->add(BOOST_CLASS_TEST_CASE(
        &ScriptVisitorTest::testVisitIf, instance));
    suite->add(BOOST_CLASS_TEST_CASE(
        &ScriptVisitorTest::testVisitWhile, instance));

    return suite;
}


ScriptVisitorTest::ScriptVisitorTest()
{
}


void ScriptVisitorTest::testVisitEmptyScript()
{
    ranally::String xml;

    xml = _algebraParser.parseString(ranally::String(""));
    _xmlParser.parse(xml)->Accept(_visitor);
    BOOST_CHECK_EQUAL(_visitor.script(), ranally::String(""));
}


void ScriptVisitorTest::testVisitName()
{
    ranally::String xml;

    xml = _algebraParser.parseString(ranally::String("a"));
    _xmlParser.parse(xml)->Accept(_visitor);
    BOOST_CHECK_EQUAL(_visitor.script(), ranally::String("a\n"));
}


void ScriptVisitorTest::testVisitAssignment()
{
    ranally::String xml;

    xml = _algebraParser.parseString(ranally::String("a = b"));
    _xmlParser.parse(xml)->Accept(_visitor);
    BOOST_CHECK_EQUAL(_visitor.script(), ranally::String("a = b\n"));
}


void ScriptVisitorTest::testVisitString()
{
    ranally::String xml;

    xml = _algebraParser.parseString(ranally::String("\"five\""));
    _xmlParser.parse(xml)->Accept(_visitor);
    BOOST_CHECK_EQUAL(_visitor.script(), ranally::String("\"five\"\n"));
}


void ScriptVisitorTest::testVisitNumber()
{
    ranally::String xml;

    xml = _algebraParser.parseString(ranally::String("5"));
    _xmlParser.parse(xml)->Accept(_visitor);
    BOOST_CHECK_EQUAL(_visitor.script(), ranally::String("5\n"));

    xml = _algebraParser.parseString(ranally::String("5L"));
    _xmlParser.parse(xml)->Accept(_visitor);
    BOOST_CHECK_EQUAL(_visitor.script(), ranally::String(
      sizeof(long) == sizeof(int64_t) ? "5\n" : "5L\n"));

    xml = _algebraParser.parseString(ranally::String("5.5"));
    _xmlParser.parse(xml)->Accept(_visitor);
    BOOST_CHECK_EQUAL(_visitor.script(), ranally::String("5.5\n"));

    // TODO add tests for all numeric types.
}


void ScriptVisitorTest::testVisitFunction()
{
    ranally::String xml;

    xml = _algebraParser.parseString(ranally::String("f()"));
    _xmlParser.parse(xml)->Accept(_visitor);
    BOOST_CHECK_EQUAL(_visitor.script(), ranally::String("f()\n"));

    xml = _algebraParser.parseString(
        ranally::String("f(1, \"2\", three, four())"));
    _xmlParser.parse(xml)->Accept(_visitor);
    BOOST_CHECK_EQUAL(_visitor.script(), ranally::String(
        "f(1, \"2\", three, four())\n"));
}


void ScriptVisitorTest::testVisitOperator()
{
    ranally::String xml;

    xml = _algebraParser.parseString(ranally::String("-a"));
    _xmlParser.parse(xml)->Accept(_visitor);
    BOOST_CHECK_EQUAL(_visitor.script(), ranally::String("-(a)\n"));

    xml = _algebraParser.parseString(ranally::String("a + b"));
    _xmlParser.parse(xml)->Accept(_visitor);
    BOOST_CHECK_EQUAL(_visitor.script(), ranally::String("(a) + (b)\n"));

    xml = _algebraParser.parseString(ranally::String("-(a + b)"));
    _xmlParser.parse(xml)->Accept(_visitor);
    BOOST_CHECK_EQUAL(_visitor.script(), ranally::String("-((a) + (b))\n"));

    xml = _algebraParser.parseString(ranally::String("a + b * c + d"));
    _xmlParser.parse(xml)->Accept(_visitor);
    BOOST_CHECK_EQUAL(_visitor.script(), ranally::String(
        "((a) + ((b) * (c))) + (d)\n"));
}


void ScriptVisitorTest::testVisitMultipleStatements()
{
    ranally::String xml;

    xml = _algebraParser.parseString(ranally::String("a\nb"));
    _xmlParser.parse(xml)->Accept(_visitor);
    BOOST_CHECK_EQUAL(_visitor.script(), ranally::String("a\nb\n"));
}


void ScriptVisitorTest::testVisitIf()
{
    ranally::String xml;

    xml = _algebraParser.parseString(ranally::String(
        "if a:\n"
        "    b\n"
        "    c"));
    _xmlParser.parse(xml)->Accept(_visitor);
    BOOST_CHECK_EQUAL(_visitor.script(), ranally::String(
        "if a:\n"
        "    b\n"
        "    c\n"));

    xml = _algebraParser.parseString(ranally::String(
        "if a:\n"
        "    b\n"
        "    c\n"
        "elif d:\n"
        "    e\n"
        "    f\n"));
    _xmlParser.parse(xml)->Accept(_visitor);
    BOOST_CHECK_EQUAL(_visitor.script(), ranally::String(
        "if a:\n"
        "    b\n"
        "    c\n"
        "else:\n"
        "    if d:\n"
        "        e\n"
        "        f\n"));

    xml = _algebraParser.parseString(ranally::String(
        "if a:\n"
        "    b\n"
        "    c\n"
        "elif d:\n"
        "    e\n"
        "    f\n"
        "else:\n"
        "    g\n"
        "    h\n"));
    _xmlParser.parse(xml)->Accept(_visitor);
    BOOST_CHECK_EQUAL(_visitor.script(), ranally::String(
        "if a:\n"
        "    b\n"
        "    c\n"
        "else:\n"
        "    if d:\n"
        "        e\n"
        "        f\n"
        "    else:\n"
        "        g\n"
        "        h\n"));
}


void ScriptVisitorTest::testVisitWhile()
{
    ranally::String xml;

    xml = _algebraParser.parseString(ranally::String(
        "while a:\n"
        "    b\n"
        "    c"));
    _xmlParser.parse(xml)->Accept(_visitor);
    BOOST_CHECK_EQUAL(_visitor.script(), ranally::String(
        "while a:\n"
        "    b\n"
        "    c\n"));

    xml = _algebraParser.parseString(ranally::String(
        "while a:\n"
        "    b\n"
        "    c\n"
        "else:\n"
        "    d\n"
        "    e"));
    _xmlParser.parse(xml)->Accept(_visitor);
    BOOST_CHECK_EQUAL(_visitor.script(), ranally::String(
        "while a:\n"
        "    b\n"
        "    c\n"
        "else:\n"
        "    d\n"
        "    e\n"));
}
