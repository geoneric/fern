#define BOOST_TEST_MODULE ranally ast
#include <boost/test/unit_test.hpp>
#include "ranally/core/exception.h"
#include "ranally/core/string.h"
#include "ranally/script/algebra_parser.h"
#include "ranally/ast/xml/syntax_tree-pskel.hxx"
#include "ranally/ast/xml/xml_parser.h"


BOOST_AUTO_TEST_SUITE(xml_parser)

BOOST_AUTO_TEST_CASE(parse_string)
{
    ranally::AlgebraParser algebra_parser;
    ranally::XmlParser xml_parser;
    ranally::String xml;
    std::shared_ptr<ranally::SyntaxVertex> tree;

    {
        // Empty xml.
        xml = algebra_parser.parse_string(ranally::String(""));
        tree = xml_parser.parse_string(xml);
    }

    {
        // Name expression.
        xml = algebra_parser.parse_string(ranally::String("a"));
        tree = xml_parser.parse_string(xml);
    }

    {
        // String expression.
        xml = algebra_parser.parse_string(ranally::String("\"five\""));
        tree = xml_parser.parse_string(xml);
    }

    {
        // TODO test all numeric types.

        // Numeric expression.
        xml = algebra_parser.parse_string(ranally::String("5"));
        tree = xml_parser.parse_string(xml);

        xml = algebra_parser.parse_string(ranally::String("5L"));
        tree = xml_parser.parse_string(xml);

        // TODO test unsigned types.

        xml = algebra_parser.parse_string(ranally::String("5.5"));
        tree = xml_parser.parse_string(xml);
    }

    {
        // Function call.
        xml = algebra_parser.parse_string(ranally::String("f()"));
        tree = xml_parser.parse_string(xml);

        xml = algebra_parser.parse_string(ranally::String(
            "f(1, \"2\", three, four())"));
        tree = xml_parser.parse_string(xml);
    }

    {
        // Operator.
        // Unary.
        xml = algebra_parser.parse_string(ranally::String("-a"));
        tree = xml_parser.parse_string(xml);

        // Binary.
        xml = algebra_parser.parse_string(ranally::String("a + b"));
        tree = xml_parser.parse_string(xml);

        // Boolean.
        xml = algebra_parser.parse_string(ranally::String("a and b"));
        tree = xml_parser.parse_string(xml);

        // Comparison.
        xml = algebra_parser.parse_string(ranally::String("a == b"));
        tree = xml_parser.parse_string(xml);
    }

    {
        // Assignment statement.
        xml = algebra_parser.parse_string(ranally::String("a = b"));
        tree = xml_parser.parse_string(xml);
    }

    {
        // Multiple statements.
        xml = algebra_parser.parse_string(ranally::String("a\nb"));
        tree = xml_parser.parse_string(xml);
    }

    {
        // If statement.
        xml = algebra_parser.parse_string(ranally::String(
            "if a:\n"
            "    b"));
        tree = xml_parser.parse_string(xml);

        xml = algebra_parser.parse_string(ranally::String(
            "if a:\n"
            "    b\n"
            "else:\n"
            "    c"));
        tree = xml_parser.parse_string(xml);

        xml = algebra_parser.parse_string(ranally::String(
            "if a:\n"
            "    b\n"
            "elif c:\n"
            "    d"));
        tree = xml_parser.parse_string(xml);
    }

    {
        // While statement.
        xml = algebra_parser.parse_string(ranally::String(
            "while a:\n"
            "    b"));
        tree = xml_parser.parse_string(xml);

        xml = algebra_parser.parse_string(ranally::String(
            "while a:\n"
            "    b\n"
            "else:\n"
            "    c"));
        tree = xml_parser.parse_string(xml);
    }

    {
        // Random string.
        BOOST_CHECK_THROW(xml_parser.parse_string(ranally::String("blabla")),
            ranally::detail::ParseError);

        // Attribute value missing.
        xml =
            "<?xml version=\"1.0\"?>"
            "<Ranally>"
              "<Expression col=\"0\">"
                "<Name>a</Name>"
              "</Expression>"
            "</Ranally>";
        BOOST_CHECK_THROW(xml_parser.parse_string(xml),
            ranally::detail::ParseError);

        // Attribute value out of range.
        xml =
            "<?xml version=\"1.0\"?>"
            "<Ranally>"
              "<Expression line=\"-1\" col=\"0\">"
                "<Name>a</Name>"
              "</Expression>"
            "</Ranally>";
        BOOST_CHECK_THROW(xml_parser.parse_string(xml),
            ranally::detail::ParseError);
    }

    {
        // Slice expression.
        xml = algebra_parser.parse_string(ranally::String("a[b]"));
        tree = xml_parser.parse_string(xml);
    }

    {
        // Function definition.
        xml = 
            "<?xml version=\"1.0\"?>"
            "<Ranally source=\"&lt;string&gt;\">"
              "<Statements>"
                "<Statement line=\"1\" col=\"0\">"
                  "<FunctionDefinition>"
                    "<Name>foo</Name>"
                    "<Expressions>"
                      "<Expression line=\"1\" col=\"8\">"
                        "<Name>a</Name>"
                      "</Expression>"
                      "<Expression line=\"1\" col=\"11\">"
                        "<Name>b</Name>"
                      "</Expression>"
                    "</Expressions>"
                    "<Statements>"
                      "<Statement line=\"2\" col=\"4\">"
                        "<Assignment>"
                          "<Expression line=\"2\" col=\"4\">"
                            "<Name>c</Name>"
                          "</Expression>"
                          "<Expression line=\"2\" col=\"8\">"
                            "<Operator>"
                              "<Name>add</Name>"
                              "<Expressions>"
                                "<Expression line=\"2\" col=\"8\">"
                                  "<Name>a</Name>"
                                "</Expression>"
                                "<Expression line=\"2\" col=\"12\">"
                                  "<Name>b</Name>"
                                "</Expression>"
                              "</Expressions>"
                            "</Operator>"
                          "</Expression>"
                        "</Assignment>"
                      "</Statement>"
                    "</Statements>"
                  "</FunctionDefinition>"
                "</Statement>"
              "</Statements>"
            "</Ranally>";
        tree = xml_parser.parse_string(xml);
    }
}

BOOST_AUTO_TEST_SUITE_END()
