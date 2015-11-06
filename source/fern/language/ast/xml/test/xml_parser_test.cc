// -----------------------------------------------------------------------------
// Fern © Geoneric
//
// This file is part of Geoneric Fern which is available under the terms of
// the GNU General Public License (GPL), version 2. If you do not want to
// be bound by the terms of the GPL, you may purchase a proprietary license
// from Geoneric (http://www.geoneric.eu/contact).
// -----------------------------------------------------------------------------
#define BOOST_TEST_MODULE fern ast xml_parser
#include <boost/test/unit_test.hpp>
#include "fern/core/exception.h"
#include "fern/language/script/algebra_parser.h"
#include "fern/language/ast/xml/syntax_tree-pskel.hxx"
#include "fern/language/ast/xml/xml_parser.h"


namespace fl = fern::language;


BOOST_AUTO_TEST_CASE(parse_string)
{
    fl::AlgebraParser algebra_parser;
    fl::XmlParser xml_parser;
    std::string xml;
    // std::shared_ptr<fl::AstVertex> tree;

    {
        // Empty xml.
        xml = algebra_parser.parse_string("");
        BOOST_CHECK_NO_THROW(xml_parser.parse_string(xml));
    }

    {
        // Name expression.
        xml = algebra_parser.parse_string("a");
        BOOST_CHECK_NO_THROW(xml_parser.parse_string(xml));
    }

    {
        // String expression.
        xml = algebra_parser.parse_string("\"five\"");
        BOOST_CHECK_NO_THROW(xml_parser.parse_string(xml));
    }

    {
        // TODO test all numeric types.

        // Numeric expression.
        xml = algebra_parser.parse_string("5");
        BOOST_CHECK_NO_THROW(xml_parser.parse_string(xml));

        xml = algebra_parser.parse_string("5L");
        BOOST_CHECK_NO_THROW(xml_parser.parse_string(xml));

        // TODO test unsigned types.

        xml = algebra_parser.parse_string("5.5");
        BOOST_CHECK_NO_THROW(xml_parser.parse_string(xml));
    }

    {
        // Function call.
        xml = algebra_parser.parse_string("f()");
        BOOST_CHECK_NO_THROW(xml_parser.parse_string(xml));

        xml = algebra_parser.parse_string(
            "f(1, \"2\", three, four())");
        BOOST_CHECK_NO_THROW(xml_parser.parse_string(xml));
    }

    {
        // Operator.
        // Unary.
        xml = algebra_parser.parse_string("-a");
        BOOST_CHECK_NO_THROW(xml_parser.parse_string(xml));

        // Binary.
        xml = algebra_parser.parse_string("a + b");
        BOOST_CHECK_NO_THROW(xml_parser.parse_string(xml));

        // Boolean.
        xml = algebra_parser.parse_string("a and b");
        BOOST_CHECK_NO_THROW(xml_parser.parse_string(xml));

        // Comparison.
        xml = algebra_parser.parse_string("a == b");
        BOOST_CHECK_NO_THROW(xml_parser.parse_string(xml));
    }

    {
        // Assignment statement.
        xml = algebra_parser.parse_string("a = b");
        BOOST_CHECK_NO_THROW(xml_parser.parse_string(xml));
    }

    {
        // Multiple statements.
        xml = algebra_parser.parse_string("a\nb");
        BOOST_CHECK_NO_THROW(xml_parser.parse_string(xml));
    }

    {
        // If statement.
        xml = algebra_parser.parse_string(
            "if a:\n"
            "    b");
        BOOST_CHECK_NO_THROW(xml_parser.parse_string(xml));

        xml = algebra_parser.parse_string(
            "if a:\n"
            "    b\n"
            "else:\n"
            "    c");
        BOOST_CHECK_NO_THROW(xml_parser.parse_string(xml));

        xml = algebra_parser.parse_string(
            "if a:\n"
            "    b\n"
            "elif c:\n"
            "    d");
        BOOST_CHECK_NO_THROW(xml_parser.parse_string(xml));
    }

    {
        // While statement.
        xml = algebra_parser.parse_string(
            "while a:\n"
            "    b");
        BOOST_CHECK_NO_THROW(xml_parser.parse_string(xml));

        xml = algebra_parser.parse_string(
            "while a:\n"
            "    b\n"
            "else:\n"
            "    c");
        BOOST_CHECK_NO_THROW(xml_parser.parse_string(xml));
    }

    {
        // Random string.
        BOOST_CHECK_THROW(xml_parser.parse_string("blabla"),
            fern::detail::ParseError);

        // Attribute value missing.
        xml =
            "<?xml version=\"1.0\"?>"
            "<Fern>"
              "<Expression col=\"0\">"
                "<Name>a</Name>"
              "</Expression>"
            "</Fern>";
        BOOST_CHECK_THROW(xml_parser.parse_string(xml),
            fern::detail::ParseError);

        // Attribute value out of range.
        xml =
            "<?xml version=\"1.0\"?>"
            "<Fern>"
              "<Expression line=\"-1\" col=\"0\">"
                "<Name>a</Name>"
              "</Expression>"
            "</Fern>";
        BOOST_CHECK_THROW(xml_parser.parse_string(xml),
            fern::detail::ParseError);
    }

    {
        // Slice expression.
        xml = algebra_parser.parse_string("a[b]");
        BOOST_CHECK_NO_THROW(xml_parser.parse_string(xml));
    }

    {
        // Attribute expression.
        xml = algebra_parser.parse_string("a.b");
        BOOST_CHECK_NO_THROW(xml_parser.parse_string(xml));
    }

    {
        // Return statement.
        xml =
            "<?xml version=\"1.0\"?>"
            "<Fern source=\"&lt;string&gt;\">"
              "<Statements>"
                "<Statement line=\"1\" col=\"0\">"
                  "<Return/>"
                "</Statement>"
              "</Statements>"
            "</Fern>";
        BOOST_CHECK_NO_THROW(xml_parser.parse_string(xml));

        xml =
            "<?xml version=\"1.0\"?>"
            "<Fern source=\"&lt;string&gt;\">"
              "<Statements>"
                "<Statement line=\"1\" col=\"0\">"
                  "<Return>"
                    "<Expression line=\"1\" col=\"7\">"
                      "<Name>c</Name>"
                    "</Expression>"
                  "</Return>"
                "</Statement>"
              "</Statements>"
            "</Fern>";
        BOOST_CHECK_NO_THROW(xml_parser.parse_string(xml));
    }

    {
        // Function definition.
        xml =
            "<?xml version=\"1.0\"?>"
            "<Fern source=\"&lt;string&gt;\">"
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
            "</Fern>";
        BOOST_CHECK_NO_THROW(xml_parser.parse_string(xml));
    }
}
