#define BOOST_TEST_MODULE geoneric ast
#include <boost/test/unit_test.hpp>
#include "geoneric/core/exception.h"
#include "geoneric/core/string.h"
#include "geoneric/script/algebra_parser.h"
#include "geoneric/ast/xml/syntax_tree-pskel.hxx"
#include "geoneric/ast/xml/xml_parser.h"


BOOST_AUTO_TEST_SUITE(xml_parser)

BOOST_AUTO_TEST_CASE(parse_string)
{
    geoneric::AlgebraParser algebra_parser;
    geoneric::XmlParser xml_parser;
    geoneric::String xml;
    // std::shared_ptr<geoneric::AstVertex> tree;

    {
        // Empty xml.
        xml = algebra_parser.parse_string(geoneric::String(""));
        BOOST_CHECK_NO_THROW(xml_parser.parse_string(xml));
    }

    {
        // Name expression.
        xml = algebra_parser.parse_string(geoneric::String("a"));
        BOOST_CHECK_NO_THROW(xml_parser.parse_string(xml));
    }

    {
        // String expression.
        xml = algebra_parser.parse_string(geoneric::String("\"five\""));
        BOOST_CHECK_NO_THROW(xml_parser.parse_string(xml));
    }

    {
        // TODO test all numeric types.

        // Numeric expression.
        xml = algebra_parser.parse_string(geoneric::String("5"));
        BOOST_CHECK_NO_THROW(xml_parser.parse_string(xml));

        xml = algebra_parser.parse_string(geoneric::String("5L"));
        BOOST_CHECK_NO_THROW(xml_parser.parse_string(xml));

        // TODO test unsigned types.

        xml = algebra_parser.parse_string(geoneric::String("5.5"));
        BOOST_CHECK_NO_THROW(xml_parser.parse_string(xml));
    }

    {
        // Function call.
        xml = algebra_parser.parse_string(geoneric::String("f()"));
        BOOST_CHECK_NO_THROW(xml_parser.parse_string(xml));

        xml = algebra_parser.parse_string(geoneric::String(
            "f(1, \"2\", three, four())"));
        BOOST_CHECK_NO_THROW(xml_parser.parse_string(xml));
    }

    {
        // Operator.
        // Unary.
        xml = algebra_parser.parse_string(geoneric::String("-a"));
        BOOST_CHECK_NO_THROW(xml_parser.parse_string(xml));

        // Binary.
        xml = algebra_parser.parse_string(geoneric::String("a + b"));
        BOOST_CHECK_NO_THROW(xml_parser.parse_string(xml));

        // Boolean.
        xml = algebra_parser.parse_string(geoneric::String("a and b"));
        BOOST_CHECK_NO_THROW(xml_parser.parse_string(xml));

        // Comparison.
        xml = algebra_parser.parse_string(geoneric::String("a == b"));
        BOOST_CHECK_NO_THROW(xml_parser.parse_string(xml));
    }

    {
        // Assignment statement.
        xml = algebra_parser.parse_string(geoneric::String("a = b"));
        BOOST_CHECK_NO_THROW(xml_parser.parse_string(xml));
    }

    {
        // Multiple statements.
        xml = algebra_parser.parse_string(geoneric::String("a\nb"));
        BOOST_CHECK_NO_THROW(xml_parser.parse_string(xml));
    }

    {
        // If statement.
        xml = algebra_parser.parse_string(geoneric::String(
            "if a:\n"
            "    b"));
        BOOST_CHECK_NO_THROW(xml_parser.parse_string(xml));

        xml = algebra_parser.parse_string(geoneric::String(
            "if a:\n"
            "    b\n"
            "else:\n"
            "    c"));
        BOOST_CHECK_NO_THROW(xml_parser.parse_string(xml));

        xml = algebra_parser.parse_string(geoneric::String(
            "if a:\n"
            "    b\n"
            "elif c:\n"
            "    d"));
        BOOST_CHECK_NO_THROW(xml_parser.parse_string(xml));
    }

    {
        // While statement.
        xml = algebra_parser.parse_string(geoneric::String(
            "while a:\n"
            "    b"));
        BOOST_CHECK_NO_THROW(xml_parser.parse_string(xml));

        xml = algebra_parser.parse_string(geoneric::String(
            "while a:\n"
            "    b\n"
            "else:\n"
            "    c"));
        BOOST_CHECK_NO_THROW(xml_parser.parse_string(xml));
    }

    {
        // Random string.
        BOOST_CHECK_THROW(xml_parser.parse_string(geoneric::String("blabla")),
            geoneric::detail::ParseError);

        // Attribute value missing.
        xml =
            "<?xml version=\"1.0\"?>"
            "<Geoneric>"
              "<Expression col=\"0\">"
                "<Name>a</Name>"
              "</Expression>"
            "</Geoneric>";
        BOOST_CHECK_THROW(xml_parser.parse_string(xml),
            geoneric::detail::ParseError);

        // Attribute value out of range.
        xml =
            "<?xml version=\"1.0\"?>"
            "<Geoneric>"
              "<Expression line=\"-1\" col=\"0\">"
                "<Name>a</Name>"
              "</Expression>"
            "</Geoneric>";
        BOOST_CHECK_THROW(xml_parser.parse_string(xml),
            geoneric::detail::ParseError);
    }

    {
        // Slice expression.
        xml = algebra_parser.parse_string(geoneric::String("a[b]"));
        BOOST_CHECK_NO_THROW(xml_parser.parse_string(xml));
    }

    {
        // Attribute expression.
        xml = algebra_parser.parse_string(geoneric::String("a.b"));
        BOOST_CHECK_NO_THROW(xml_parser.parse_string(xml));
    }

    {
        // Return statement.
        xml =
            "<?xml version=\"1.0\"?>"
            "<Geoneric source=\"&lt;string&gt;\">"
              "<Statements>"
                "<Statement line=\"1\" col=\"0\">"
                  "<Return/>"
                "</Statement>"
              "</Statements>"
            "</Geoneric>";
        BOOST_CHECK_NO_THROW(xml_parser.parse_string(xml));

        xml =
            "<?xml version=\"1.0\"?>"
            "<Geoneric source=\"&lt;string&gt;\">"
              "<Statements>"
                "<Statement line=\"1\" col=\"0\">"
                  "<Return>"
                    "<Expression line=\"1\" col=\"7\">"
                      "<Name>c</Name>"
                    "</Expression>"
                  "</Return>"
                "</Statement>"
              "</Statements>"
            "</Geoneric>";
        BOOST_CHECK_NO_THROW(xml_parser.parse_string(xml));
    }

    {
        // Function definition.
        xml =
            "<?xml version=\"1.0\"?>"
            "<Geoneric source=\"&lt;string&gt;\">"
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
            "</Geoneric>";
        BOOST_CHECK_NO_THROW(xml_parser.parse_string(xml));
    }
}

BOOST_AUTO_TEST_SUITE_END()
