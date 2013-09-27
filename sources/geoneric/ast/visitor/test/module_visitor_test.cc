#define BOOST_TEST_MODULE geoneric ast
#include <boost/test/unit_test.hpp>
#include "geoneric/core/string.h"
#include "geoneric/script/algebra_parser.h"
#include "geoneric/ast/core/module_vertex.h"
#include "geoneric/ast/visitor/module_visitor.h"
#include "geoneric/ast/xml/xml_parser.h"


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

    geoneric::AlgebraParser _algebra_parser;

    geoneric::XmlParser _xml_parser;

    geoneric::ModuleVisitor _visitor;

};


BOOST_FIXTURE_TEST_SUITE(module_visitor, Support)


BOOST_AUTO_TEST_CASE(visit_empty_module)
{
    geoneric::String xml;

    xml = _algebra_parser.parse_string(geoneric::String(""));
    _xml_parser.parse_string(xml)->Accept(_visitor);
    BOOST_CHECK_EQUAL(_visitor.module(), geoneric::String(""));
}


BOOST_AUTO_TEST_CASE(visit_name)
{
    geoneric::String xml;

    xml = _algebra_parser.parse_string(geoneric::String("a"));
    _xml_parser.parse_string(xml)->Accept(_visitor);
    BOOST_CHECK_EQUAL(_visitor.module(), geoneric::String("a\n"));
}


BOOST_AUTO_TEST_CASE(visit_assignment)
{
    geoneric::String xml;

    xml = _algebra_parser.parse_string(geoneric::String("a = b"));
    _xml_parser.parse_string(xml)->Accept(_visitor);
    BOOST_CHECK_EQUAL(_visitor.module(), geoneric::String("a = b\n"));
}


BOOST_AUTO_TEST_CASE(visit_string)
{
    geoneric::String xml;

    xml = _algebra_parser.parse_string(geoneric::String("\"five\""));
    _xml_parser.parse_string(xml)->Accept(_visitor);
    BOOST_CHECK_EQUAL(_visitor.module(), geoneric::String("\"five\"\n"));
}


BOOST_AUTO_TEST_CASE(visit_number)
{
    geoneric::String xml;

    xml = _algebra_parser.parse_string(geoneric::String("5"));
    _xml_parser.parse_string(xml)->Accept(_visitor);
    BOOST_CHECK_EQUAL(_visitor.module(), geoneric::String("5\n"));

    xml = _algebra_parser.parse_string(geoneric::String("5L"));
    _xml_parser.parse_string(xml)->Accept(_visitor);
    BOOST_CHECK_EQUAL(_visitor.module(), geoneric::String(
      sizeof(long) == sizeof(int64_t) ? "5\n" : "5L\n"));

    xml = _algebra_parser.parse_string(geoneric::String("5.5"));
    _xml_parser.parse_string(xml)->Accept(_visitor);
    BOOST_CHECK_EQUAL(_visitor.module(), geoneric::String("5.5\n"));

    // TODO add tests for all numeric types.
}


BOOST_AUTO_TEST_CASE(visit_function)
{
    geoneric::String xml;

    xml = _algebra_parser.parse_string(geoneric::String("f()"));
    _xml_parser.parse_string(xml)->Accept(_visitor);
    BOOST_CHECK_EQUAL(_visitor.module(), geoneric::String("f()\n"));

    xml = _algebra_parser.parse_string(
        geoneric::String("f(1, \"2\", three, four())"));
    _xml_parser.parse_string(xml)->Accept(_visitor);
    BOOST_CHECK_EQUAL(_visitor.module(), geoneric::String(
        "f(1, \"2\", three, four())\n"));
}


BOOST_AUTO_TEST_CASE(visit_operator)
{
    geoneric::String xml;

    xml = _algebra_parser.parse_string(geoneric::String("-a"));
    _xml_parser.parse_string(xml)->Accept(_visitor);
    BOOST_CHECK_EQUAL(_visitor.module(), geoneric::String("-(a)\n"));

    xml = _algebra_parser.parse_string(geoneric::String("a + b"));
    _xml_parser.parse_string(xml)->Accept(_visitor);
    BOOST_CHECK_EQUAL(_visitor.module(), geoneric::String("(a) + (b)\n"));

    xml = _algebra_parser.parse_string(geoneric::String("-(a + b)"));
    _xml_parser.parse_string(xml)->Accept(_visitor);
    BOOST_CHECK_EQUAL(_visitor.module(), geoneric::String("-((a) + (b))\n"));

    xml = _algebra_parser.parse_string(geoneric::String("a + b * c + d"));
    _xml_parser.parse_string(xml)->Accept(_visitor);
    BOOST_CHECK_EQUAL(_visitor.module(), geoneric::String(
        "((a) + ((b) * (c))) + (d)\n"));
}


BOOST_AUTO_TEST_CASE(visit_multiple_statements)
{
    geoneric::String xml;

    xml = _algebra_parser.parse_string(geoneric::String("a\nb"));
    _xml_parser.parse_string(xml)->Accept(_visitor);
    BOOST_CHECK_EQUAL(_visitor.module(), geoneric::String("a\nb\n"));
}


BOOST_AUTO_TEST_CASE(visit_if)
{
    geoneric::String xml;

    xml = _algebra_parser.parse_string(geoneric::String(
        "if a:\n"
        "    b\n"
        "    c"));
    _xml_parser.parse_string(xml)->Accept(_visitor);
    BOOST_CHECK_EQUAL(_visitor.module(), geoneric::String(
        "if a:\n"
        "    b\n"
        "    c\n"));

    xml = _algebra_parser.parse_string(geoneric::String(
        "if a:\n"
        "    b\n"
        "    c\n"
        "elif d:\n"
        "    e\n"
        "    f\n"));
    _xml_parser.parse_string(xml)->Accept(_visitor);
    BOOST_CHECK_EQUAL(_visitor.module(), geoneric::String(
        "if a:\n"
        "    b\n"
        "    c\n"
        "else:\n"
        "    if d:\n"
        "        e\n"
        "        f\n"));

    xml = _algebra_parser.parse_string(geoneric::String(
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
    BOOST_CHECK_EQUAL(_visitor.module(), geoneric::String(
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
    geoneric::String xml;

    xml = _algebra_parser.parse_string(geoneric::String(
        "while a:\n"
        "    b\n"
        "    c"));
    _xml_parser.parse_string(xml)->Accept(_visitor);
    BOOST_CHECK_EQUAL(_visitor.module(), geoneric::String(
        "while a:\n"
        "    b\n"
        "    c\n"));

    xml = _algebra_parser.parse_string(geoneric::String(
        "while a:\n"
        "    b\n"
        "    c\n"
        "else:\n"
        "    d\n"
        "    e"));
    _xml_parser.parse_string(xml)->Accept(_visitor);
    BOOST_CHECK_EQUAL(_visitor.module(), geoneric::String(
        "while a:\n"
        "    b\n"
        "    c\n"
        "else:\n"
        "    d\n"
        "    e\n"));
}


BOOST_AUTO_TEST_CASE(visit_subscript)
{
    geoneric::String xml;

    xml = _algebra_parser.parse_string(geoneric::String(
        "a = b[c]"));
    std::cout << xml << std::endl;
    _xml_parser.parse_string(xml)->Accept(_visitor);
    BOOST_CHECK_EQUAL(_visitor.module(), geoneric::String(
        "a = (b)[c]\n"));

    xml = _algebra_parser.parse_string(geoneric::String(
        "a = (b + c)[c > d]"));
    _xml_parser.parse_string(xml)->Accept(_visitor);
    BOOST_CHECK_EQUAL(_visitor.module(), geoneric::String(
        "a = ((b) + (c))[(c) > (d)]\n"));
}


BOOST_AUTO_TEST_CASE(visit_attribute)
{
    geoneric::String xml;

    xml = _algebra_parser.parse_string(geoneric::String(
        "a = b.c"));
    std::cout << xml << std::endl;
    _xml_parser.parse_string(xml)->Accept(_visitor);
    BOOST_CHECK_EQUAL(_visitor.module(), geoneric::String(
        "a = (b).c\n"));

    xml = _algebra_parser.parse_string(geoneric::String(
        "a = (b + c).d"));
    _xml_parser.parse_string(xml)->Accept(_visitor);
    BOOST_CHECK_EQUAL(_visitor.module(), geoneric::String(
        "a = ((b) + (c)).d\n"));
}

BOOST_AUTO_TEST_SUITE_END()
