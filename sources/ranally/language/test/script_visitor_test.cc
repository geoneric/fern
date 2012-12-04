#define BOOST_TEST_MODULE ranally language
#include <boost/test/unit_test.hpp>
#include "ranally/core/string.h"
#include "ranally/language/algebra_parser.h"
#include "ranally/language/script_vertex.h"
#include "ranally/language/script_visitor.h"
#include "ranally/language/xml_parser.h"


class Support
{

public:

    Support()
        : _algebra_parser(),
          _xml_parser(),
          _visitor()
    {
    }

protected:

    ranally::AlgebraParser _algebra_parser;

    ranally::XmlParser _xml_parser;

    ranally::ScriptVisitor _visitor;

};


BOOST_FIXTURE_TEST_SUITE(script_visitor, Support)


BOOST_AUTO_TEST_CASE(visit_empty_script)
{
    ranally::String xml;

    xml = _algebra_parser.parse_string(ranally::String(""));
    _xml_parser.parse_string(xml)->Accept(_visitor);
    BOOST_CHECK_EQUAL(_visitor.script(), ranally::String(""));
}


BOOST_AUTO_TEST_CASE(visit_name)
{
    ranally::String xml;

    xml = _algebra_parser.parse_string(ranally::String("a"));
    _xml_parser.parse_string(xml)->Accept(_visitor);
    BOOST_CHECK_EQUAL(_visitor.script(), ranally::String("a\n"));
}


BOOST_AUTO_TEST_CASE(visit_assignment)
{
    ranally::String xml;

    xml = _algebra_parser.parse_string(ranally::String("a = b"));
    _xml_parser.parse_string(xml)->Accept(_visitor);
    BOOST_CHECK_EQUAL(_visitor.script(), ranally::String("a = b\n"));
}


BOOST_AUTO_TEST_CASE(visit_string)
{
    ranally::String xml;

    xml = _algebra_parser.parse_string(ranally::String("\"five\""));
    _xml_parser.parse_string(xml)->Accept(_visitor);
    BOOST_CHECK_EQUAL(_visitor.script(), ranally::String("\"five\"\n"));
}


BOOST_AUTO_TEST_CASE(visit_number)
{
    ranally::String xml;

    xml = _algebra_parser.parse_string(ranally::String("5"));
    _xml_parser.parse_string(xml)->Accept(_visitor);
    BOOST_CHECK_EQUAL(_visitor.script(), ranally::String("5\n"));

    xml = _algebra_parser.parse_string(ranally::String("5L"));
    _xml_parser.parse_string(xml)->Accept(_visitor);
    BOOST_CHECK_EQUAL(_visitor.script(), ranally::String(
      sizeof(long) == sizeof(int64_t) ? "5\n" : "5L\n"));

    xml = _algebra_parser.parse_string(ranally::String("5.5"));
    _xml_parser.parse_string(xml)->Accept(_visitor);
    BOOST_CHECK_EQUAL(_visitor.script(), ranally::String("5.5\n"));

    // TODO add tests for all numeric types.
}


BOOST_AUTO_TEST_CASE(visit_function)
{
    ranally::String xml;

    xml = _algebra_parser.parse_string(ranally::String("f()"));
    _xml_parser.parse_string(xml)->Accept(_visitor);
    BOOST_CHECK_EQUAL(_visitor.script(), ranally::String("f()\n"));

    xml = _algebra_parser.parse_string(
        ranally::String("f(1, \"2\", three, four())"));
    _xml_parser.parse_string(xml)->Accept(_visitor);
    BOOST_CHECK_EQUAL(_visitor.script(), ranally::String(
        "f(1, \"2\", three, four())\n"));
}


BOOST_AUTO_TEST_CASE(visit_operator)
{
    ranally::String xml;

    xml = _algebra_parser.parse_string(ranally::String("-a"));
    _xml_parser.parse_string(xml)->Accept(_visitor);
    BOOST_CHECK_EQUAL(_visitor.script(), ranally::String("-(a)\n"));

    xml = _algebra_parser.parse_string(ranally::String("a + b"));
    _xml_parser.parse_string(xml)->Accept(_visitor);
    BOOST_CHECK_EQUAL(_visitor.script(), ranally::String("(a) + (b)\n"));

    xml = _algebra_parser.parse_string(ranally::String("-(a + b)"));
    _xml_parser.parse_string(xml)->Accept(_visitor);
    BOOST_CHECK_EQUAL(_visitor.script(), ranally::String("-((a) + (b))\n"));

    xml = _algebra_parser.parse_string(ranally::String("a + b * c + d"));
    _xml_parser.parse_string(xml)->Accept(_visitor);
    BOOST_CHECK_EQUAL(_visitor.script(), ranally::String(
        "((a) + ((b) * (c))) + (d)\n"));
}


BOOST_AUTO_TEST_CASE(visit_multiple_statements)
{
    ranally::String xml;

    xml = _algebra_parser.parse_string(ranally::String("a\nb"));
    _xml_parser.parse_string(xml)->Accept(_visitor);
    BOOST_CHECK_EQUAL(_visitor.script(), ranally::String("a\nb\n"));
}


BOOST_AUTO_TEST_CASE(visit_if)
{
    ranally::String xml;

    xml = _algebra_parser.parse_string(ranally::String(
        "if a:\n"
        "    b\n"
        "    c"));
    _xml_parser.parse_string(xml)->Accept(_visitor);
    BOOST_CHECK_EQUAL(_visitor.script(), ranally::String(
        "if a:\n"
        "    b\n"
        "    c\n"));

    xml = _algebra_parser.parse_string(ranally::String(
        "if a:\n"
        "    b\n"
        "    c\n"
        "elif d:\n"
        "    e\n"
        "    f\n"));
    _xml_parser.parse_string(xml)->Accept(_visitor);
    BOOST_CHECK_EQUAL(_visitor.script(), ranally::String(
        "if a:\n"
        "    b\n"
        "    c\n"
        "else:\n"
        "    if d:\n"
        "        e\n"
        "        f\n"));

    xml = _algebra_parser.parse_string(ranally::String(
        "if a:\n"
        "    b\n"
        "    c\n"
        "elif d:\n"
        "    e\n"
        "    f\n"
        "else:\n"
        "    g\n"
        "    h\n"));
    _xml_parser.parse_string(xml)->Accept(_visitor);
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


BOOST_AUTO_TEST_CASE(visit_while)
{
    ranally::String xml;

    xml = _algebra_parser.parse_string(ranally::String(
        "while a:\n"
        "    b\n"
        "    c"));
    _xml_parser.parse_string(xml)->Accept(_visitor);
    BOOST_CHECK_EQUAL(_visitor.script(), ranally::String(
        "while a:\n"
        "    b\n"
        "    c\n"));

    xml = _algebra_parser.parse_string(ranally::String(
        "while a:\n"
        "    b\n"
        "    c\n"
        "else:\n"
        "    d\n"
        "    e"));
    _xml_parser.parse_string(xml)->Accept(_visitor);
    BOOST_CHECK_EQUAL(_visitor.script(), ranally::String(
        "while a:\n"
        "    b\n"
        "    c\n"
        "else:\n"
        "    d\n"
        "    e\n"));
}

BOOST_AUTO_TEST_SUITE_END()
