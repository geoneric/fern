#define BOOST_TEST_MODULE fern script
#include <boost/test/unit_test.hpp>
#include "fern/core/exception.h"
#include "fern/core/string.h"
#include "fern/script/algebra_parser.h"


BOOST_AUTO_TEST_SUITE(algebra_parser)

BOOST_AUTO_TEST_CASE(parse_empty_script)
{
    fern::AlgebraParser parser;

    {
        fern::String xml(parser.parse_string(fern::String("")));
        BOOST_CHECK_EQUAL(xml, fern::String(
            "<?xml version=\"1.0\"?>"
            "<Geoneric source=\"&lt;string&gt;\">"
                "<Statements/>"
            "</Geoneric>"));
    }
}


BOOST_AUTO_TEST_CASE(parse_name)
{
    fern::AlgebraParser parser;

    {
        fern::String xml(parser.parse_string(fern::String("a")));
        BOOST_CHECK_EQUAL(xml, fern::String(
          "<?xml version=\"1.0\"?>"
          "<Geoneric source=\"&lt;string&gt;\">"
            "<Statements>"
              "<Statement line=\"1\" col=\"0\">"
                "<Expression line=\"1\" col=\"0\">"
                  "<Name>a</Name>"
                "</Expression>"
              "</Statement>"
            "</Statements>"
          "</Geoneric>"));
    }

    // TODO #if PYTHONVER >= 2.7/3.0?
    // {
    //     fern::String xml(parser.parse_string(fern::String("ñ")));
    //     BOOST_CHECK_EQUAL(xml, fern::String(
    //       "<?xml version=\"1.0\"?>"
    //       "<Geoneric>"
    //         "<Statements>"
    //           "<Statement>"
    //             "<Expression line=\"1\" col=\"0\">"
    //               "<Name>ñ</Name>"
    //             "</Expression>"
    //           "</Statement>"
    //         "</Statements>"
    //       "</Geoneric>"));
    // }
}


BOOST_AUTO_TEST_CASE(parse_assignment)
{
    fern::AlgebraParser parser;

    {
        fern::String xml(parser.parse_string(fern::String("a = b")));
        BOOST_CHECK_EQUAL(xml, fern::String(
            "<?xml version=\"1.0\"?>"
            "<Geoneric source=\"&lt;string&gt;\">"
              "<Statements>"
                "<Statement line=\"1\" col=\"0\">"
                  "<Assignment>"
                    "<Expression line=\"1\" col=\"0\">"
                      "<Name>a</Name>"
                    "</Expression>"
                    "<Expression line=\"1\" col=\"4\">"
                      "<Name>b</Name>"
                    "</Expression>"
                  "</Assignment>"
                "</Statement>"
              "</Statements>"
            "</Geoneric>"));
    }
}


BOOST_AUTO_TEST_CASE(parse_string)
{
    fern::AlgebraParser parser;

    {
        fern::String xml(parser.parse_string(fern::String("\"five\"")));
        BOOST_CHECK_EQUAL(xml, fern::String(
            "<?xml version=\"1.0\"?>"
            "<Geoneric source=\"&lt;string&gt;\">"
              "<Statements>"
                "<Statement line=\"1\" col=\"0\">"
                  "<Expression line=\"1\" col=\"0\">"
                    "<String>five</String>"
                  "</Expression>"
                "</Statement>"
              "</Statements>"
            "</Geoneric>"));
    }

    {
        fern::String xml(parser.parse_string(fern::String("\"\"")));
        BOOST_CHECK_EQUAL(xml, fern::String(
            "<?xml version=\"1.0\"?>"
            "<Geoneric source=\"&lt;string&gt;\">"
              "<Statements>"
                "<Statement line=\"1\" col=\"0\">"
                  "<Expression line=\"1\" col=\"0\">"
                    "<String/>"
                  "</Expression>"
                "</Statement>"
              "</Statements>"
            "</Geoneric>"));
    }

    {
        fern::String xml(parser.parse_string(fern::String("\" \"")));
        BOOST_CHECK_EQUAL(xml, fern::String(
            "<?xml version=\"1.0\"?>"
            "<Geoneric source=\"&lt;string&gt;\">"
              "<Statements>"
                "<Statement line=\"1\" col=\"0\">"
                  "<Expression line=\"1\" col=\"0\">"
                    "<String> </String>"
                  "</Expression>"
                "</Statement>"
              "</Statements>"
            "</Geoneric>"));
    }

    // Test handling of Unicode characters.
    {
        fern::String xml(parser.parse_string(fern::String("\"mañana\"")));
        BOOST_CHECK_EQUAL(xml, fern::String(
            "<?xml version=\"1.0\"?>"
            "<Geoneric source=\"&lt;string&gt;\">"
              "<Statements>"
                "<Statement line=\"1\" col=\"0\">"
                  "<Expression line=\"1\" col=\"0\">"
                    "<String>mañana</String>"
                  "</Expression>"
                "</Statement>"
              "</Statements>"
            "</Geoneric>"));
    }

    {
        BOOST_CHECK_THROW(parser.parse_string(fern::String("if")),
            fern::detail::ParseError);
    }

    {
        BOOST_CHECK_THROW(parser.parse_string(fern::String("yield")),
            fern::detail::UnsupportedLanguageConstruct);
    }

    {
        BOOST_CHECK_THROW(parser.parse_string(fern::String("print(5)")),
            fern::detail::UnsupportedLanguageConstruct);
    }
}


BOOST_AUTO_TEST_CASE(parse_number)
{
    fern::AlgebraParser parser;

    {
        fern::String xml(parser.parse_string(fern::String("5")));
        BOOST_CHECK_EQUAL(xml, fern::String(boost::format(
            "<?xml version=\"1.0\"?>"
            "<Geoneric source=\"&lt;string&gt;\">"
              "<Statements>"
                "<Statement line=\"1\" col=\"0\">"
                  "<Expression line=\"1\" col=\"0\">"
                    "<Number>"
                      "<Integer>"
                        "<Size>%1%</Size>"
                        "<Value>5</Value>"
                      "</Integer>"
                    "</Number>"
                  "</Expression>"
                "</Statement>"
              "</Statements>"
            "</Geoneric>") % (sizeof(long) * 8)));
    }

    {
        fern::String xml(parser.parse_string(fern::String("5L")));
        BOOST_CHECK_EQUAL(xml, fern::String(
            "<?xml version=\"1.0\"?>"
            "<Geoneric source=\"&lt;string&gt;\">"
              "<Statements>"
                "<Statement line=\"1\" col=\"0\">"
                  "<Expression line=\"1\" col=\"0\">"
                    "<Number>"
                      "<Integer>"
                        "<Size>64</Size>"
                        "<Value>5</Value>"
                      "</Integer>"
                    "</Number>"
                  "</Expression>"
                "</Statement>"
              "</Statements>"
            "</Geoneric>"));
    }

    {
        fern::String xml(parser.parse_string(fern::String("5.5")));
        BOOST_CHECK_EQUAL(xml, fern::String(
            "<?xml version=\"1.0\"?>"
            "<Geoneric source=\"&lt;string&gt;\">"
              "<Statements>"
                "<Statement line=\"1\" col=\"0\">"
                  "<Expression line=\"1\" col=\"0\">"
                    "<Number>"
                      "<Float>"
                        "<Size>64</Size>"
                        "<Value>5.5</Value>"
                      "</Float>"
                    "</Number>"
                  "</Expression>"
                "</Statement>"
              "</Statements>"
            "</Geoneric>"));
    }
}


BOOST_AUTO_TEST_CASE(parse_call)
{
    fern::AlgebraParser parser;
    fern::String xml;

    {
        xml = parser.parse_string(fern::String("f()"));
        BOOST_CHECK_EQUAL(xml, fern::String(
            "<?xml version=\"1.0\"?>"
            "<Geoneric source=\"&lt;string&gt;\">"
              "<Statements>"
                "<Statement line=\"1\" col=\"0\">"
                  "<Expression line=\"1\" col=\"0\">"
                    "<FunctionCall>"
                      "<Name>f</Name>"
                      "<Expressions/>"
                    "</FunctionCall>"
                  "</Expression>"
                "</Statement>"
              "</Statements>"
            "</Geoneric>"));
    }

    {
        xml = parser.parse_string(fern::String("f(1, \"2\", three, four())"));
        BOOST_CHECK_EQUAL(xml, fern::String(boost::format(
            "<?xml version=\"1.0\"?>"
            "<Geoneric source=\"&lt;string&gt;\">"
              "<Statements>"
                "<Statement line=\"1\" col=\"0\">"
                  "<Expression line=\"1\" col=\"0\">"
                    "<FunctionCall>"
                      "<Name>f</Name>"
                      "<Expressions>"
                        "<Expression line=\"1\" col=\"2\">"
                          "<Number>"
                            "<Integer>"
                              "<Size>%1%</Size>"
                              "<Value>1</Value>"
                            "</Integer>"
                          "</Number>"
                        "</Expression>"
                        "<Expression line=\"1\" col=\"5\">"
                          "<String>2</String>"
                        "</Expression>"
                        "<Expression line=\"1\" col=\"10\">"
                          "<Name>three</Name>"
                        "</Expression>"
                        "<Expression line=\"1\" col=\"17\">"
                          "<FunctionCall>"
                            "<Name>four</Name>"
                            "<Expressions/>"
                          "</FunctionCall>"
                        "</Expression>"
                      "</Expressions>"
                    "</FunctionCall>"
                  "</Expression>"
                "</Statement>"
              "</Statements>"
            "</Geoneric>") % (sizeof(long) * 8)));
    }
}


// BOOST_AUTO_TEST_CASE(parse_print)
// {
//   fern::AlgebraParser parser;
//   fern::String xml;
// 
//   {
//     xml = parser.parse_string(fern::String("print"));
//     BOOST_CHECK_EQUAL(xml, fern::String(
//       "<?xml version=\"1.0\"?>"
//       "<Geoneric source=\"&lt;string&gt;\">"
//         "<Statements>"
//           "<Statement>"
//             "<Expression line=\"1\" col=\"0\">"
//               "<FunctionCall>"
//                 "<Name>print</Name>"
//                 "<Expressions/>"
//               "</FunctionCall>"
//             "</Expression>"
//           "</Statement>"
//         "</Statements>"
//       "</Geoneric>"));
//   }
// 
//   // {
//   //   xml = parser.parse_string(fern::String("print(1, \"2\", three, four())"));
//   //   BOOST_CHECK_EQUAL(xml, fern::String(boost::format(
//   //     "<?xml version=\"1.0\"?>"
//   //     "<Geoneric source=\"&lt;string&gt;\">"
//   //       "<Statements>"
//   //         "<Statement>"
//   //           "<Expression line=\"1\" col=\"0\">"
//   //             "<FunctionCall>"
//   //               "<Name>f</Name>"
//   //               "<Expressions>"
//   //                 "<Expression line=\"1\" col=\"6\">"
//   //                   "<Number>"
//   //                     "<Integer>"
//   //                       "<Size>%1%</Size>"
//   //                       "<Value>1</Value>"
//   //                     "</Integer>"
//   //                   "</Number>"
//   //                 "</Expression>"
//   //                 "<Expression line=\"1\" col=\"9\">"
//   //                   "<String>2</String>"
//   //                 "</Expression>"
//   //                 "<Expression line=\"1\" col=\"14\">"
//   //                   "<Name>three</Name>"
//   //                 "</Expression>"
//   //                 "<Expression line=\"1\" col=\"21\">"
//   //                   "<FunctionCall>"
//   //                     "<Name>four</Name>"
//   //                     "<Expressions/>"
//   //                   "</FunctionCall>"
//   //                 "</Expression>"
//   //               "</Expressions>"
//   //             "</FunctionCall>"
//   //           "</Expression>"
//   //         "</Statement>"
//   //       "</Statements>"
//   //     "</Geoneric>") % (sizeof(long) * 8)));
//   // }
// }


BOOST_AUTO_TEST_CASE(parse_unary_operator)
{
    fern::AlgebraParser parser;
    fern::String xml;

    {
        xml = parser.parse_string(fern::String("-a"));
        BOOST_CHECK_EQUAL(xml, fern::String(
            "<?xml version=\"1.0\"?>"
            "<Geoneric source=\"&lt;string&gt;\">"
              "<Statements>"
                "<Statement line=\"1\" col=\"0\">"
                  "<Expression line=\"1\" col=\"0\">"
                    "<Operator>"
                      "<Name>UnarySub</Name>"
                      "<Expressions>"
                        "<Expression line=\"1\" col=\"1\">"
                          "<Name>a</Name>"
                        "</Expression>"
                      "</Expressions>"
                    "</Operator>"
                  "</Expression>"
                "</Statement>"
              "</Statements>"
            "</Geoneric>"));
    }
}


BOOST_AUTO_TEST_CASE(parse_binary_operator)
{
    fern::AlgebraParser parser;
    fern::String xml;

    {
        xml = parser.parse_string(fern::String("a + b"));
        BOOST_CHECK_EQUAL(xml, fern::String(
            "<?xml version=\"1.0\"?>"
            "<Geoneric source=\"&lt;string&gt;\">"
              "<Statements>"
                "<Statement line=\"1\" col=\"0\">"
                  "<Expression line=\"1\" col=\"0\">"
                    "<Operator>"
                      "<Name>add</Name>"
                      "<Expressions>"
                        "<Expression line=\"1\" col=\"0\">"
                          "<Name>a</Name>"
                        "</Expression>"
                        "<Expression line=\"1\" col=\"4\">"
                          "<Name>b</Name>"
                        "</Expression>"
                      "</Expressions>"
                    "</Operator>"
                  "</Expression>"
                "</Statement>"
              "</Statements>"
            "</Geoneric>"));
    }
}


BOOST_AUTO_TEST_CASE(parse_boolean_operator)
{
    fern::AlgebraParser parser;
    fern::String xml;

    {
        xml = parser.parse_string(fern::String("a and b"));
        BOOST_CHECK_EQUAL(xml, fern::String(
            "<?xml version=\"1.0\"?>"
            "<Geoneric source=\"&lt;string&gt;\">"
              "<Statements>"
                "<Statement line=\"1\" col=\"0\">"
                  "<Expression line=\"1\" col=\"0\">"
                    "<Operator>"
                      "<Name>And</Name>"
                      "<Expressions>"
                        "<Expression line=\"1\" col=\"0\">"
                          "<Name>a</Name>"
                        "</Expression>"
                        "<Expression line=\"1\" col=\"6\">"
                          "<Name>b</Name>"
                        "</Expression>"
                      "</Expressions>"
                    "</Operator>"
                  "</Expression>"
                "</Statement>"
              "</Statements>"
            "</Geoneric>"));
    }
}


BOOST_AUTO_TEST_CASE(parse_comparison_operator)
{
    fern::AlgebraParser parser;
    fern::String xml;

    {
        xml = parser.parse_string(fern::String("a <= b"));
        BOOST_CHECK_EQUAL(xml, fern::String(
            "<?xml version=\"1.0\"?>"
            "<Geoneric source=\"&lt;string&gt;\">"
              "<Statements>"
                "<Statement line=\"1\" col=\"0\">"
                  "<Expression line=\"1\" col=\"0\">"
                    "<Operator>"
                      "<Name>LtE</Name>"
                      "<Expressions>"
                        "<Expression line=\"1\" col=\"0\">"
                          "<Name>a</Name>"
                        "</Expression>"
                        "<Expression line=\"1\" col=\"5\">"
                          "<Name>b</Name>"
                        "</Expression>"
                      "</Expressions>"
                    "</Operator>"
                  "</Expression>"
                "</Statement>"
              "</Statements>"
            "</Geoneric>"));
    }
}


BOOST_AUTO_TEST_CASE(parse_multiple_statements)
{
    fern::AlgebraParser parser;

    {
        fern::String xml(parser.parse_string(fern::String("a\nb")));
        BOOST_CHECK_EQUAL(xml, fern::String(
            "<?xml version=\"1.0\"?>"
            "<Geoneric source=\"&lt;string&gt;\">"
              "<Statements>"
                "<Statement line=\"1\" col=\"0\">"
                  "<Expression line=\"1\" col=\"0\">"
                    "<Name>a</Name>"
                  "</Expression>"
                "</Statement>"
                "<Statement line=\"2\" col=\"0\">"
                  "<Expression line=\"2\" col=\"0\">"
                    "<Name>b</Name>"
                  "</Expression>"
                "</Statement>"
              "</Statements>"
            "</Geoneric>"));
    }
}


BOOST_AUTO_TEST_CASE(parse_if)
{
    fern::AlgebraParser parser;
    fern::String xml;

    {
        xml = parser.parse_string(fern::String(
            "if a:\n"
            "  b"));
        BOOST_CHECK_EQUAL(xml, fern::String(
            "<?xml version=\"1.0\"?>"
            "<Geoneric source=\"&lt;string&gt;\">"
              "<Statements>"
                "<Statement line=\"1\" col=\"0\">"
                  "<If>"
                    "<Expression line=\"1\" col=\"3\">"
                      "<Name>a</Name>"
                    "</Expression>"
                    "<Statements>"
                      "<Statement line=\"2\" col=\"2\">"
                        "<Expression line=\"2\" col=\"2\">"
                          "<Name>b</Name>"
                        "</Expression>"
                      "</Statement>"
                    "</Statements>"
                    "<Statements/>"
                  "</If>"
                "</Statement>"
              "</Statements>"
            "</Geoneric>"));
    }

    {
        xml = parser.parse_string(fern::String(
            "if a:\n"
            "  b\n"
            "elif(c):\n"
            "  d"));
        BOOST_CHECK_EQUAL(xml, fern::String(
            "<?xml version=\"1.0\"?>"
            "<Geoneric source=\"&lt;string&gt;\">"
              "<Statements>"
                "<Statement line=\"1\" col=\"0\">"
                  "<If>"
                    "<Expression line=\"1\" col=\"3\">"
                      "<Name>a</Name>"
                    "</Expression>"
                    "<Statements>"
                      "<Statement line=\"2\" col=\"2\">"
                        "<Expression line=\"2\" col=\"2\">"
                          "<Name>b</Name>"
                        "</Expression>"
                      "</Statement>"
                    "</Statements>"
                    "<Statements>"
                      "<Statement line=\"3\" col=\"4\">"
                        "<If>"
                          "<Expression line=\"3\" col=\"5\">"
                            "<Name>c</Name>"
                          "</Expression>"
                          "<Statements>"
                            "<Statement line=\"4\" col=\"2\">"
                              "<Expression line=\"4\" col=\"2\">"
                                "<Name>d</Name>"
                              "</Expression>"
                            "</Statement>"
                          "</Statements>"
                          "<Statements/>"
                        "</If>"
                      "</Statement>"
                    "</Statements>"
                  "</If>"
                "</Statement>"
              "</Statements>"
            "</Geoneric>"));
    }

    {
        xml = parser.parse_string(fern::String(
            "if a:\n"
            "  b\n"
            "elif c:\n"
            "  d\n"
            "else:\n"
            "  e"));
        BOOST_CHECK_EQUAL(xml, fern::String(
            "<?xml version=\"1.0\"?>"
            "<Geoneric source=\"&lt;string&gt;\">"
              "<Statements>"
                "<Statement line=\"1\" col=\"0\">"
                  "<If>"
                    "<Expression line=\"1\" col=\"3\">"
                      "<Name>a</Name>"
                    "</Expression>"
                    "<Statements>"
                      "<Statement line=\"2\" col=\"2\">"
                        "<Expression line=\"2\" col=\"2\">"
                          "<Name>b</Name>"
                        "</Expression>"
                      "</Statement>"
                    "</Statements>"
                    "<Statements>"
                      "<Statement line=\"3\" col=\"5\">"
                        "<If>"
                          "<Expression line=\"3\" col=\"5\">"
                            "<Name>c</Name>"
                          "</Expression>"
                          "<Statements>"
                            "<Statement line=\"4\" col=\"2\">"
                              "<Expression line=\"4\" col=\"2\">"
                                "<Name>d</Name>"
                              "</Expression>"
                            "</Statement>"
                          "</Statements>"
                          "<Statements>"
                            "<Statement line=\"6\" col=\"2\">"
                              "<Expression line=\"6\" col=\"2\">"
                                "<Name>e</Name>"
                              "</Expression>"
                            "</Statement>"
                          "</Statements>"
                        "</If>"
                      "</Statement>"
                    "</Statements>"
                  "</If>"
                "</Statement>"
              "</Statements>"
            "</Geoneric>"));
    }
}


BOOST_AUTO_TEST_CASE(parse_while)
{
    fern::AlgebraParser parser;
    fern::String xml;

    {
        xml = parser.parse_string(fern::String(
            "while a:\n"
            "  b"));
        BOOST_CHECK_EQUAL(xml, fern::String(
            "<?xml version=\"1.0\"?>"
            "<Geoneric source=\"&lt;string&gt;\">"
              "<Statements>"
                "<Statement line=\"1\" col=\"0\">"
                  "<While>"
                    "<Expression line=\"1\" col=\"6\">"
                      "<Name>a</Name>"
                    "</Expression>"
                    "<Statements>"
                      "<Statement line=\"2\" col=\"2\">"
                        "<Expression line=\"2\" col=\"2\">"
                          "<Name>b</Name>"
                        "</Expression>"
                      "</Statement>"
                    "</Statements>"
                    "<Statements/>"
                  "</While>"
                "</Statement>"
              "</Statements>"
            "</Geoneric>"));
    }

    {
        xml = parser.parse_string(fern::String(
            "while a:\n"
            "  b\n"
            "else:\n"
            "  c"));
        BOOST_CHECK_EQUAL(xml, fern::String(
            "<?xml version=\"1.0\"?>"
            "<Geoneric source=\"&lt;string&gt;\">"
              "<Statements>"
                "<Statement line=\"1\" col=\"0\">"
                  "<While>"
                    "<Expression line=\"1\" col=\"6\">"
                      "<Name>a</Name>"
                    "</Expression>"
                    "<Statements>"
                      "<Statement line=\"2\" col=\"2\">"
                        "<Expression line=\"2\" col=\"2\">"
                          "<Name>b</Name>"
                        "</Expression>"
                      "</Statement>"
                    "</Statements>"
                    "<Statements>"
                      "<Statement line=\"4\" col=\"2\">"
                        "<Expression line=\"4\" col=\"2\">"
                          "<Name>c</Name>"
                        "</Expression>"
                      "</Statement>"
                    "</Statements>"
                  "</While>"
                "</Statement>"
              "</Statements>"
            "</Geoneric>"));
    }
}


BOOST_AUTO_TEST_CASE(parse_file)
{
    fern::AlgebraParser parser;
    fern::String filename;

    {
        filename = "DoesNotExist.ran";
        BOOST_CHECK_THROW(parser.parse_file(filename),
            fern::detail::FileOpenError);
    }
}


BOOST_AUTO_TEST_CASE(parse_slice)
{
    fern::AlgebraParser parser;
    fern::String xml;

    {
        xml = parser.parse_string(fern::String(
            "a[b]"));
        BOOST_CHECK_EQUAL(xml, fern::String(
            "<?xml version=\"1.0\"?>"
            "<Geoneric source=\"&lt;string&gt;\">"
              "<Statements>"
                "<Statement line=\"1\" col=\"0\">"
                  "<Expression line=\"1\" col=\"0\">"
                    "<Subscript>"
                      "<Expression line=\"1\" col=\"0\">"
                        "<Name>a</Name>"
                      "</Expression>"
                      "<Expression line=\"1\" col=\"2\">"
                        "<Name>b</Name>"
                      "</Expression>"
                    "</Subscript>"
                  "</Expression>"
                "</Statement>"
              "</Statements>"
            "</Geoneric>"));
    }
}


BOOST_AUTO_TEST_CASE(parse_attribute)
{
    fern::AlgebraParser parser;
    fern::String xml;

    {
        xml = parser.parse_string(fern::String(
            "a.b"));
        BOOST_CHECK_EQUAL(xml, fern::String(
            "<?xml version=\"1.0\"?>"
            "<Geoneric source=\"&lt;string&gt;\">"
              "<Statements>"
                "<Statement line=\"1\" col=\"0\">"
                  "<Expression line=\"1\" col=\"0\">"
                    "<Attribute>"
                      "<Expression line=\"1\" col=\"0\">"
                        "<Name>a</Name>"
                      "</Expression>"
                      "<Name>b</Name>"
                    "</Attribute>"
                  "</Expression>"
                "</Statement>"
              "</Statements>"
            "</Geoneric>"));
    }
}


BOOST_AUTO_TEST_CASE(return_)
{
    fern::AlgebraParser parser;
    fern::String xml;

    {
        xml = parser.parse_string(fern::String(
            "return"));
        BOOST_CHECK_EQUAL(xml, fern::String(
            "<?xml version=\"1.0\"?>"
            "<Geoneric source=\"&lt;string&gt;\">"
              "<Statements>"
                "<Statement line=\"1\" col=\"0\">"
                  "<Return/>"
                "</Statement>"
              "</Statements>"
            "</Geoneric>"));
    }

    {
        xml = parser.parse_string(fern::String(
            "return c"));
        BOOST_CHECK_EQUAL(xml, fern::String(
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
            "</Geoneric>"));
    }
}


BOOST_AUTO_TEST_CASE(parse_function)
{
    fern::AlgebraParser parser;
    fern::String xml;

    {
        xml = parser.parse_string(fern::String(
            "def foo(a, b):\n"
            "    c = a + b\n"));
        BOOST_CHECK_EQUAL(xml, fern::String(
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
            "</Geoneric>"));
    }
}

BOOST_AUTO_TEST_SUITE_END()
