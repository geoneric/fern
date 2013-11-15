#define BOOST_TEST_MODULE geoneric ast
#include <boost/test/unit_test.hpp>
#include "fern/core/string.h"
#include "fern/script/algebra_parser.h"
#include "fern/ast/core/module_vertex.h"
#include "fern/ast/visitor/module_visitor.h"
#include "fern/ast/xml/xml_parser.h"


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

    fern::AlgebraParser _algebra_parser;

    fern::XmlParser _xml_parser;

    fern::ModuleVisitor _visitor;

};


BOOST_FIXTURE_TEST_SUITE(module_visitor, Support)


BOOST_AUTO_TEST_CASE(visit_empty_module)
{
    fern::String xml;

    xml = _algebra_parser.parse_string(fern::String(""));
    _xml_parser.parse_string(xml)->Accept(_visitor);
    BOOST_CHECK_EQUAL(_visitor.module(), fern::String(""));
}


BOOST_AUTO_TEST_CASE(visit_name)
{
    fern::String xml;

    xml = _algebra_parser.parse_string(fern::String("a"));
    _xml_parser.parse_string(xml)->Accept(_visitor);
    BOOST_CHECK_EQUAL(_visitor.module(), fern::String("a\n"));
}


BOOST_AUTO_TEST_CASE(visit_assignment)
{
    fern::String xml;

    xml = _algebra_parser.parse_string(fern::String("a = b"));
    _xml_parser.parse_string(xml)->Accept(_visitor);
    BOOST_CHECK_EQUAL(_visitor.module(), fern::String("a = b\n"));
}


BOOST_AUTO_TEST_CASE(visit_string)
{
    fern::String xml;

    xml = _algebra_parser.parse_string(fern::String("\"five\""));
    _xml_parser.parse_string(xml)->Accept(_visitor);
    BOOST_CHECK_EQUAL(_visitor.module(), fern::String("\"five\"\n"));
}


BOOST_AUTO_TEST_CASE(visit_number)
{
    fern::String xml;

    xml = _algebra_parser.parse_string(fern::String("5"));
    _xml_parser.parse_string(xml)->Accept(_visitor);
    BOOST_CHECK_EQUAL(_visitor.module(), fern::String("5\n"));

    xml = _algebra_parser.parse_string(fern::String("5L"));
    _xml_parser.parse_string(xml)->Accept(_visitor);
    BOOST_CHECK_EQUAL(_visitor.module(), fern::String(
      sizeof(long) == sizeof(int64_t) ? "5\n" : "5L\n"));

    xml = _algebra_parser.parse_string(fern::String("5.5"));
    _xml_parser.parse_string(xml)->Accept(_visitor);
    BOOST_CHECK_EQUAL(_visitor.module(), fern::String("5.5\n"));

    // TODO add tests for all numeric types.
}


BOOST_AUTO_TEST_CASE(visit_function)
{
    fern::String xml;

    xml = _algebra_parser.parse_string(fern::String("f()"));
    _xml_parser.parse_string(xml)->Accept(_visitor);
    BOOST_CHECK_EQUAL(_visitor.module(), fern::String("f()\n"));

    xml = _algebra_parser.parse_string(
        fern::String("f(1, \"2\", three, four())"));
    _xml_parser.parse_string(xml)->Accept(_visitor);
    BOOST_CHECK_EQUAL(_visitor.module(), fern::String(
        "f(1, \"2\", three, four())\n"));
}


BOOST_AUTO_TEST_CASE(visit_operator)
{
    fern::String xml;

    xml = _algebra_parser.parse_string(fern::String("-a"));
    _xml_parser.parse_string(xml)->Accept(_visitor);
    BOOST_CHECK_EQUAL(_visitor.module(), fern::String("-(a)\n"));

    xml = _algebra_parser.parse_string(fern::String("a + b"));
    _xml_parser.parse_string(xml)->Accept(_visitor);
    BOOST_CHECK_EQUAL(_visitor.module(), fern::String("(a) + (b)\n"));

    xml = _algebra_parser.parse_string(fern::String("-(a + b)"));
    _xml_parser.parse_string(xml)->Accept(_visitor);
    BOOST_CHECK_EQUAL(_visitor.module(), fern::String("-((a) + (b))\n"));

    xml = _algebra_parser.parse_string(fern::String("a + b * c + d"));
    _xml_parser.parse_string(xml)->Accept(_visitor);
    BOOST_CHECK_EQUAL(_visitor.module(), fern::String(
        "((a) + ((b) * (c))) + (d)\n"));
}


BOOST_AUTO_TEST_CASE(visit_multiple_statements)
{
    fern::String xml;

    xml = _algebra_parser.parse_string(fern::String("a\nb"));
    _xml_parser.parse_string(xml)->Accept(_visitor);
    BOOST_CHECK_EQUAL(_visitor.module(), fern::String("a\nb\n"));
}


BOOST_AUTO_TEST_CASE(visit_if)
{
    fern::String xml;

    xml = _algebra_parser.parse_string(fern::String(
        "if a:\n"
        "    b\n"
        "    c"));
    _xml_parser.parse_string(xml)->Accept(_visitor);
    BOOST_CHECK_EQUAL(_visitor.module(), fern::String(
        "if a:\n"
        "    b\n"
        "    c\n"));

    xml = _algebra_parser.parse_string(fern::String(
        "if a:\n"
        "    b\n"
        "    c\n"
        "elif d:\n"
        "    e\n"
        "    f\n"));
    _xml_parser.parse_string(xml)->Accept(_visitor);
    BOOST_CHECK_EQUAL(_visitor.module(), fern::String(
        "if a:\n"
        "    b\n"
        "    c\n"
        "else:\n"
        "    if d:\n"
        "        e\n"
        "        f\n"));

    xml = _algebra_parser.parse_string(fern::String(
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
    BOOST_CHECK_EQUAL(_visitor.module(), fern::String(
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
    fern::String xml;

    xml = _algebra_parser.parse_string(fern::String(
        "while a:\n"
        "    b\n"
        "    c"));
    _xml_parser.parse_string(xml)->Accept(_visitor);
    BOOST_CHECK_EQUAL(_visitor.module(), fern::String(
        "while a:\n"
        "    b\n"
        "    c\n"));

    xml = _algebra_parser.parse_string(fern::String(
        "while a:\n"
        "    b\n"
        "    c\n"
        "else:\n"
        "    d\n"
        "    e"));
    _xml_parser.parse_string(xml)->Accept(_visitor);
    BOOST_CHECK_EQUAL(_visitor.module(), fern::String(
        "while a:\n"
        "    b\n"
        "    c\n"
        "else:\n"
        "    d\n"
        "    e\n"));
}


BOOST_AUTO_TEST_CASE(visit_subscript)
{
    fern::String xml;

    xml = _algebra_parser.parse_string(fern::String(
        "a = b[c]"));
    std::cout << xml << std::endl;
    _xml_parser.parse_string(xml)->Accept(_visitor);
    BOOST_CHECK_EQUAL(_visitor.module(), fern::String(
        "a = (b)[c]\n"));

    xml = _algebra_parser.parse_string(fern::String(
        "a = (b + c)[c > d]"));
    _xml_parser.parse_string(xml)->Accept(_visitor);
    BOOST_CHECK_EQUAL(_visitor.module(), fern::String(
        "a = ((b) + (c))[(c) > (d)]\n"));
}


BOOST_AUTO_TEST_CASE(visit_attribute)
{
    fern::String xml;

    xml = _algebra_parser.parse_string(fern::String(
        "a = b.c"));
    std::cout << xml << std::endl;
    _xml_parser.parse_string(xml)->Accept(_visitor);
    BOOST_CHECK_EQUAL(_visitor.module(), fern::String(
        "a = (b).c\n"));

    xml = _algebra_parser.parse_string(fern::String(
        "a = (b + c).d"));
    _xml_parser.parse_string(xml)->Accept(_visitor);
    BOOST_CHECK_EQUAL(_visitor.module(), fern::String(
        "a = ((b) + (c)).d\n"));
}

BOOST_AUTO_TEST_SUITE_END()
