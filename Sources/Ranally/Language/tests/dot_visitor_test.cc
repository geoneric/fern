#define BOOST_TEST_MODULE ranally language
#include <boost/test/included/unit_test.hpp>
#include "Ranally/Util/string.h"
#include "Ranally/Language/algebra_parser.h"
#include "Ranally/Language/script_vertex.h"
#include "Ranally/Language/xml_parser.h"


class Support
{

public:

    Support()
        : _algebraParser(),
          _xmlParser()
    {
    }

protected:

    ranally::AlgebraParser _algebraParser;

    ranally::XmlParser _xmlParser;

};


BOOST_FIXTURE_TEST_SUITE(dot_visitor, Support)

BOOST_AUTO_TEST_CASE(visit_empty_script)
{
    // ranally::String xml;

    // xml = _algebraParser.parseString(ranally::String(""));
    // BOOST_CHECK(_xmlParser.parse(xml)->Accept(_visitor) ==
    //   "digraph G {\n"
    //   "}\n"
    // );
}


BOOST_AUTO_TEST_CASE(visit_name)
{
    // ranally::String xml;

    // xml = _algebraParser.parseString(ranally::String("a"));
    // BOOST_CHECK(_xmlParser.parse(xml)->Accept(_visitor) ==
    //   "digraph G {\n"
    //   "  a;\n"
    //   "}\n"
    // );
}


BOOST_AUTO_TEST_CASE(visit_assignment)
{
    // ranally::String xml;

    // xml = _algebraParser.parseString(ranally::String("a = b"));
    // BOOST_CHECK(_xmlParser.parse(xml)->Accept(_visitor) ==
    //   "digraph G {\n"
    //   "  b -> a;\n"
    //   "}\n"
    // );
}


BOOST_AUTO_TEST_CASE(visit_string)
{
    // ranally::String xml;

    // xml = _algebraParser.parseString(ranally::String("\"five\""));
    // BOOST_CHECK(_xmlParser.parse(xml)->Accept(_visitor) ==
    //   "digraph G {\n"
    //   "  \"five\";\n"
    //   "}\n"
    // );
}


BOOST_AUTO_TEST_CASE(visit_number)
{
    // ranally::String xml;

    // xml = _algebraParser.parseString(ranally::String("5"));
    // BOOST_CHECK(_xmlParser.parse(xml)->Accept(_visitor) ==
    //   "digraph G {\n"
    //   "  5;\n"
    //   "}\n"
    // );

    // xml = _algebraParser.parseString(ranally::String("5L"));
    // BOOST_CHECK(_xmlParser.parse(xml)->Accept(_visitor) ==
    //   "digraph G {\n"
    //   "  5;\n"
    //   "}\n"
    // );

    // xml = _algebraParser.parseString(ranally::String("5.5"));
    // BOOST_CHECK(_xmlParser.parse(xml)->Accept(_visitor) ==
    //   "digraph G {\n"
    //   "  5.5;\n"
    //   "}\n"
    // );

    // // TODO add tests for all numeric types.
}


BOOST_AUTO_TEST_CASE(visit_function)
{
    // ranally::String xml;

    // xml = _algebraParser.parseString(ranally::String("f()"));
    // BOOST_CHECK(_xmlParser.parse(xml)->Accept(_visitor) ==
    //   "digraph G {\n"
    //   "  f;\n"
    //   "}\n"
    // );

    // xml = _algebraParser.parseString(ranally::String("f(1, \"2\", three, four())"));
    // BOOST_CHECK(_xmlParser.parse(xml)->Accept(_visitor) ==
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
    // ranally::String xml;

    // xml = _algebraParser.parseString(ranally::String("-a"));
    // BOOST_CHECK(_xmlParser.parse(xml)->Accept(_visitor) == "-(a)\n");

    // xml = _algebraParser.parseString(ranally::String("a + b"));
    // BOOST_CHECK(_xmlParser.parse(xml)->Accept(_visitor) == "(a) + (b)\n");

    // xml = _algebraParser.parseString(ranally::String("-(a + b)"));
    // BOOST_CHECK(_xmlParser.parse(xml)->Accept(_visitor) == "-((a) + (b))\n");

    // xml = _algebraParser.parseString(ranally::String("a + b * c + d"));
    // BOOST_CHECK(_xmlParser.parse(xml)->Accept(_visitor) ==
    //   "((a) + ((b) * (c))) + (d)\n");
}


BOOST_AUTO_TEST_CASE(visit_multiple_statements)
{
    // ranally::String xml;

    // xml = _algebraParser.parseString(ranally::String("a\nb"));
    // BOOST_CHECK(_xmlParser.parse(xml)->Accept(_visitor) == "a\nb\n");
}


BOOST_AUTO_TEST_CASE(visit_if)
{
    // ranally::String xml;

    // xml = _algebraParser.parseString(ranally::String(
    //     "if a:\n"
    //     "  b\n"
    //     "  c"));
    // BOOST_CHECK(_xmlParser.parse(xml)->Accept(_visitor) ==
    //     "if a:\n"
    //     "  b\n"
    //     "  c\n");

    // xml = _algebraParser.parseString(ranally::String(
    //     "if a:\n"
    //     "  b\n"
    //     "  c\n"
    //     "elif d:\n"
    //     "  e\n"
    //     "  f\n"));
    // BOOST_CHECK(_xmlParser.parse(xml)->Accept(_visitor) ==
    //     "if a:\n"
    //     "  b\n"
    //     "  c\n"
    //     "else:\n"
    //     "  if d:\n"
    //     "    e\n"
    //     "    f\n");

    // xml = _algebraParser.parseString(ranally::String(
    //     "if a:\n"
    //     "  b\n"
    //     "  c\n"
    //     "elif d:\n"
    //     "  e\n"
    //     "  f\n"
    //     "else:\n"
    //     "  g\n"
    //     "  h\n"));
    // BOOST_CHECK(_xmlParser.parse(xml)->Accept(_visitor) ==
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
    // ranally::String xml;

    // xml = _algebraParser.parseString(ranally::String(
    //     "while a:\n"
    //     "  b\n"
    //     "  c"));
    // BOOST_CHECK(_xmlParser.parse(xml)->Accept(_visitor) ==
    //     "while a:\n"
    //     "  b\n"
    //     "  c\n");

    // xml = _algebraParser.parseString(ranally::String(
    //     "while a:\n"
    //     "  b\n"
    //     "  c\n"
    //     "else:\n"
    //     "  d\n"
    //     "  e"));
    // BOOST_CHECK(_xmlParser.parse(xml)->Accept(_visitor) ==
    //     "while a:\n"
    //     "  b\n"
    //     "  c\n"
    //     "else:\n"
    //     "  d\n"
    //     "  e\n");
}

BOOST_AUTO_TEST_SUITE_END()
