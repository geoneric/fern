// -----------------------------------------------------------------------------
// Fern © Geoneric
//
// This file is part of Geoneric Fern which is available under the terms of
// the GNU General Public License (GPL), version 2. If you do not want to
// be bound by the terms of the GPL, you may purchase a proprietary license
// from Geoneric (http://www.geoneric.eu/contact).
// -----------------------------------------------------------------------------
#define BOOST_TEST_MODULE fern ast
#include <boost/test/unit_test.hpp>
#include "fern/language/script/algebra_parser.h"
#include "fern/language/ast/core/module_vertex.h"
#include "fern/language/ast/xml/xml_parser.h"


namespace fl = fern::language;


class Support
{

public:

    Support()
        : _algebra_parser(),
          _xml_parser()
    {
    }

protected:

    fl::AlgebraParser _algebra_parser;

    fl::XmlParser _xml_parser;

};


BOOST_FIXTURE_TEST_SUITE(dot_visitor, Support)

BOOST_AUTO_TEST_CASE(visit_empty_script)
{
    // std::string xml;

    // xml = _algebra_parser.parse_string("");
    // BOOST_CHECK(_xml_parser.parse(xml)->Accept(_visitor) ==
    //   "digraph G {\n"
    //   "}\n"
    // );
}


BOOST_AUTO_TEST_CASE(visit_name)
{
    // std::string xml;

    // xml = _algebra_parser.parse_string("a");
    // BOOST_CHECK(_xml_parser.parse(xml)->Accept(_visitor) ==
    //   "digraph G {\n"
    //   "  a;\n"
    //   "}\n"
    // );
}


BOOST_AUTO_TEST_CASE(visit_assignment)
{
    // std::string xml;

    // xml = _algebra_parser.parse_string("a = b");
    // BOOST_CHECK(_xml_parser.parse(xml)->Accept(_visitor) ==
    //   "digraph G {\n"
    //   "  b -> a;\n"
    //   "}\n"
    // );
}


BOOST_AUTO_TEST_CASE(visit_string)
{
    // std::string xml;

    // xml = _algebra_parser.parse_string("\"five\"");
    // BOOST_CHECK(_xml_parser.parse(xml)->Accept(_visitor) ==
    //   "digraph G {\n"
    //   "  \"five\";\n"
    //   "}\n"
    // );
}


BOOST_AUTO_TEST_CASE(visit_number)
{
    // std::string xml;

    // xml = _algebra_parser.parse_string("5");
    // BOOST_CHECK(_xml_parser.parse(xml)->Accept(_visitor) ==
    //   "digraph G {\n"
    //   "  5;\n"
    //   "}\n"
    // );

    // xml = _algebra_parser.parse_string("5L");
    // BOOST_CHECK(_xml_parser.parse(xml)->Accept(_visitor) ==
    //   "digraph G {\n"
    //   "  5;\n"
    //   "}\n"
    // );

    // xml = _algebra_parser.parse_string("5.5");
    // BOOST_CHECK(_xml_parser.parse(xml)->Accept(_visitor) ==
    //   "digraph G {\n"
    //   "  5.5;\n"
    //   "}\n"
    // );

    // // TODO add tests for all numeric types.
}


BOOST_AUTO_TEST_CASE(visit_function)
{
    // std::string xml;

    // xml = _algebra_parser.parse_string("f()");
    // BOOST_CHECK(_xml_parser.parse(xml)->Accept(_visitor) ==
    //   "digraph G {\n"
    //   "  f;\n"
    //   "}\n"
    // );

    // xml = _algebra_parser.parse_string("f(1, \"2\", three, four())");
    // BOOST_CHECK(_xml_parser.parse(xml)->Accept(_visitor) ==
    //   "digraph G {\n"
    //   "  1 -> f;\n"
    //   "  \"2\" -> f;\n"
    //   "  three -> f;\n"
    //   "  four -> f;\n"
    //   "}\n"
    // );
}


BOOST_AUTO_TEST_CASE(visit_operator)
{
    // std::string xml;

    // xml = _algebra_parser.parse_string("-a");
    // BOOST_CHECK(_xml_parser.parse(xml)->Accept(_visitor) == "-(a)\n");

    // xml = _algebra_parser.parse_string("a + b");
    // BOOST_CHECK(_xml_parser.parse(xml)->Accept(_visitor) == "(a) + (b)\n");

    // xml = _algebra_parser.parse_string("-(a + b)");
    // BOOST_CHECK(_xml_parser.parse(xml)->Accept(_visitor) == "-((a) + (b))\n");

    // xml = _algebra_parser.parse_string("a + b * c + d");
    // BOOST_CHECK(_xml_parser.parse(xml)->Accept(_visitor) ==
    //   "((a) + ((b) * (c))) + (d)\n");
}


BOOST_AUTO_TEST_CASE(visit_multiple_statements)
{
    // std::string xml;

    // xml = _algebra_parser.parse_string("a\nb");
    // BOOST_CHECK(_xml_parser.parse(xml)->Accept(_visitor) == "a\nb\n");
}


BOOST_AUTO_TEST_CASE(visit_if)
{
    // std::string xml;

    // xml = _algebra_parser.parse_string(
    //     "if a:\n"
    //     "  b\n"
    //     "  c");
    // BOOST_CHECK(_xml_parser.parse(xml)->Accept(_visitor) ==
    //     "if a:\n"
    //     "  b\n"
    //     "  c\n");

    // xml = _algebra_parser.parse_string(
    //     "if a:\n"
    //     "  b\n"
    //     "  c\n"
    //     "elif d:\n"
    //     "  e\n"
    //     "  f\n");
    // BOOST_CHECK(_xml_parser.parse(xml)->Accept(_visitor) ==
    //     "if a:\n"
    //     "  b\n"
    //     "  c\n"
    //     "else:\n"
    //     "  if d:\n"
    //     "    e\n"
    //     "    f\n");

    // xml = _algebra_parser.parse_string(
    //     "if a:\n"
    //     "  b\n"
    //     "  c\n"
    //     "elif d:\n"
    //     "  e\n"
    //     "  f\n"
    //     "else:\n"
    //     "  g\n"
    //     "  h\n");
    // BOOST_CHECK(_xml_parser.parse(xml)->Accept(_visitor) ==
    //     "if a:\n"
    //     "  b\n"
    //     "  c\n"
    //     "else:\n"
    //     "  if d:\n"
    //     "    e\n"
    //     "    f\n"
    //     "  else:\n"
    //     "    g\n"
    //     "    h\n");
}


BOOST_AUTO_TEST_CASE(visit_while)
{
    // std::string xml;

    // xml = _algebra_parser.parse_string(
    //     "while a:\n"
    //     "  b\n"
    //     "  c");
    // BOOST_CHECK(_xml_parser.parse(xml)->Accept(_visitor) ==
    //     "while a:\n"
    //     "  b\n"
    //     "  c\n");

    // xml = _algebra_parser.parse_string(
    //     "while a:\n"
    //     "  b\n"
    //     "  c\n"
    //     "else:\n"
    //     "  d\n"
    //     "  e");
    // BOOST_CHECK(_xml_parser.parse(xml)->Accept(_visitor) ==
    //     "while a:\n"
    //     "  b\n"
    //     "  c\n"
    //     "else:\n"
    //     "  d\n"
    //     "  e\n");
}

BOOST_AUTO_TEST_SUITE_END()
