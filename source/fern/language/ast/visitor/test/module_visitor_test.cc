// -----------------------------------------------------------------------------
// Fern © Geoneric
//
// This file is part of Geoneric Fern which is available under the terms of
// the GNU General Public License (GPL), version 2. If you do not want to
// be bound by the terms of the GPL, you may purchase a proprietary license
// from Geoneric (http://www.geoneric.eu/contact).
// -----------------------------------------------------------------------------
#define BOOST_TEST_MODULE fern ast
#include <iostream>
#include <boost/test/unit_test.hpp>
#include "fern/language/script/algebra_parser.h"
#include "fern/language/ast/core/module_vertex.h"
#include "fern/language/ast/visitor/module_visitor.h"
#include "fern/language/ast/xml/xml_parser.h"


namespace fl = fern::language;


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

    fl::AlgebraParser _algebra_parser;

    fl::XmlParser _xml_parser;

    fl::ModuleVisitor _visitor;

};


BOOST_FIXTURE_TEST_SUITE(module_visitor, Support)


BOOST_AUTO_TEST_CASE(visit_empty_module)
{
    std::string xml;

    xml = _algebra_parser.parse_string("");
    _xml_parser.parse_string(xml)->Accept(_visitor);
    BOOST_CHECK_EQUAL(_visitor.module(), "");
}


BOOST_AUTO_TEST_CASE(visit_name)
{
    std::string xml;

    xml = _algebra_parser.parse_string("a");
    _xml_parser.parse_string(xml)->Accept(_visitor);
    BOOST_CHECK_EQUAL(_visitor.module(), "a\n");
}


BOOST_AUTO_TEST_CASE(visit_assignment)
{
    std::string xml;

    xml = _algebra_parser.parse_string("a = b");
    _xml_parser.parse_string(xml)->Accept(_visitor);
    BOOST_CHECK_EQUAL(_visitor.module(), "a = b\n");
}


BOOST_AUTO_TEST_CASE(visit_string)
{
    std::string xml;

    xml = _algebra_parser.parse_string("\"five\"");
    _xml_parser.parse_string(xml)->Accept(_visitor);
    BOOST_CHECK_EQUAL(_visitor.module(), "\"five\"\n");
}


BOOST_AUTO_TEST_CASE(visit_number)
{
    std::string xml;

    xml = _algebra_parser.parse_string("5");
    _xml_parser.parse_string(xml)->Accept(_visitor);
    BOOST_CHECK_EQUAL(_visitor.module(), "5\n");

    xml = _algebra_parser.parse_string("5L");
    _xml_parser.parse_string(xml)->Accept(_visitor);
    BOOST_CHECK_EQUAL(_visitor.module(),
      sizeof(long) == sizeof(int64_t) ? "5\n" : "5L\n");

    xml = _algebra_parser.parse_string("5.5");
    _xml_parser.parse_string(xml)->Accept(_visitor);
    BOOST_CHECK_EQUAL(_visitor.module(), "5.5\n");

    // TODO add tests for all numeric types.
}


BOOST_AUTO_TEST_CASE(visit_function)
{
    std::string xml;

    xml = _algebra_parser.parse_string("f()");
    _xml_parser.parse_string(xml)->Accept(_visitor);
    BOOST_CHECK_EQUAL(_visitor.module(), "f()\n");

    xml = _algebra_parser.parse_string(
        "f(1, \"2\", three, four())");
    _xml_parser.parse_string(xml)->Accept(_visitor);
    BOOST_CHECK_EQUAL(_visitor.module(),
        "f(1, \"2\", three, four())\n");
}


BOOST_AUTO_TEST_CASE(visit_operator)
{
    std::string xml;

    xml = _algebra_parser.parse_string("-a");
    _xml_parser.parse_string(xml)->Accept(_visitor);
    BOOST_CHECK_EQUAL(_visitor.module(), "-(a)\n");

    xml = _algebra_parser.parse_string("a + b");
    _xml_parser.parse_string(xml)->Accept(_visitor);
    BOOST_CHECK_EQUAL(_visitor.module(), "(a) + (b)\n");

    xml = _algebra_parser.parse_string("-(a + b)");
    _xml_parser.parse_string(xml)->Accept(_visitor);
    BOOST_CHECK_EQUAL(_visitor.module(), "-((a) + (b))\n");

    xml = _algebra_parser.parse_string("a + b * c + d");
    _xml_parser.parse_string(xml)->Accept(_visitor);
    BOOST_CHECK_EQUAL(_visitor.module(),
        "((a) + ((b) * (c))) + (d)\n");
}


BOOST_AUTO_TEST_CASE(visit_multiple_statements)
{
    std::string xml;

    xml = _algebra_parser.parse_string("a\nb");
    _xml_parser.parse_string(xml)->Accept(_visitor);
    BOOST_CHECK_EQUAL(_visitor.module(), "a\nb\n");
}


BOOST_AUTO_TEST_CASE(visit_if)
{
    std::string xml;

    xml = _algebra_parser.parse_string(
        "if a:\n"
        "    b\n"
        "    c");
    _xml_parser.parse_string(xml)->Accept(_visitor);
    BOOST_CHECK_EQUAL(_visitor.module(),
        "if a:\n"
        "    b\n"
        "    c\n");

    xml = _algebra_parser.parse_string(
        "if a:\n"
        "    b\n"
        "    c\n"
        "elif d:\n"
        "    e\n"
        "    f\n");
    _xml_parser.parse_string(xml)->Accept(_visitor);
    BOOST_CHECK_EQUAL(_visitor.module(),
        "if a:\n"
        "    b\n"
        "    c\n"
        "else:\n"
        "    if d:\n"
        "        e\n"
        "        f\n");

    xml = _algebra_parser.parse_string(
        "if a:\n"
        "    b\n"
        "    c\n"
        "elif d:\n"
        "    e\n"
        "    f\n"
        "else:\n"
        "    g\n"
        "    h\n");
    _xml_parser.parse_string(xml)->Accept(_visitor);
    BOOST_CHECK_EQUAL(_visitor.module(),
        "if a:\n"
        "    b\n"
        "    c\n"
        "else:\n"
        "    if d:\n"
        "        e\n"
        "        f\n"
        "    else:\n"
        "        g\n"
        "        h\n");
}


BOOST_AUTO_TEST_CASE(visit_while)
{
    std::string xml;

    xml = _algebra_parser.parse_string(
        "while a:\n"
        "    b\n"
        "    c");
    _xml_parser.parse_string(xml)->Accept(_visitor);
    BOOST_CHECK_EQUAL(_visitor.module(),
        "while a:\n"
        "    b\n"
        "    c\n");

    xml = _algebra_parser.parse_string(
        "while a:\n"
        "    b\n"
        "    c\n"
        "else:\n"
        "    d\n"
        "    e");
    _xml_parser.parse_string(xml)->Accept(_visitor);
    BOOST_CHECK_EQUAL(_visitor.module(),
        "while a:\n"
        "    b\n"
        "    c\n"
        "else:\n"
        "    d\n"
        "    e\n");
}


BOOST_AUTO_TEST_CASE(visit_subscript)
{
    std::string xml;

    xml = _algebra_parser.parse_string(
        "a = b[c]");
    std::cout << xml << std::endl;
    _xml_parser.parse_string(xml)->Accept(_visitor);
    BOOST_CHECK_EQUAL(_visitor.module(),
        "a = (b)[c]\n");

    xml = _algebra_parser.parse_string(
        "a = (b + c)[c > d]");
    _xml_parser.parse_string(xml)->Accept(_visitor);
    BOOST_CHECK_EQUAL(_visitor.module(),
        "a = ((b) + (c))[(c) > (d)]\n");
}


BOOST_AUTO_TEST_CASE(visit_attribute)
{
    std::string xml;

    xml = _algebra_parser.parse_string(
        "a = b.c");
    std::cout << xml << std::endl;
    _xml_parser.parse_string(xml)->Accept(_visitor);
    BOOST_CHECK_EQUAL(_visitor.module(),
        "a = (b).c\n");

    xml = _algebra_parser.parse_string(
        "a = (b + c).d");
    _xml_parser.parse_string(xml)->Accept(_visitor);
    BOOST_CHECK_EQUAL(_visitor.module(),
        "a = ((b) + (c)).d\n");
}

BOOST_AUTO_TEST_SUITE_END()
