#define BOOST_TEST_MODULE geoneric script
#include <boost/test/unit_test.hpp>
#include "geoneric/core/exception.h"
#include "geoneric/core/string.h"
#include "geoneric/script/algebra_parser.h"


BOOST_AUTO_TEST_SUITE(algebra_parser)

BOOST_AUTO_TEST_CASE(parse_empty_script)
{
    geoneric::AlgebraParser parser;

    {
        geoneric::String xml(parser.parse_string(geoneric::String("")));
        BOOST_CHECK_EQUAL(xml, geoneric::String(
            "<?xml version=\"1.0\"?>"
            "<Geoneric source=\"&lt;string&gt;\">"
                "<Statements/>"
            "</Geoneric>"));
    }
}


BOOST_AUTO_TEST_CASE(parse_name)
{
    geoneric::AlgebraParser parser;

    {
        geoneric::String xml(parser.parse_string(geoneric::String("a")));
        BOOST_CHECK_EQUAL(xml, geoneric::String(
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
    //     geoneric::String xml(parser.parse_string(geoneric::String("ñ")));
    //     BOOST_CHECK_EQUAL(xml, geoneric::String(
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
    geoneric::AlgebraParser parser;

    {
        geoneric::String xml(parser.parse_string(geoneric::String("a = b")));
        BOOST_CHECK_EQUAL(xml, geoneric::String(
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
    geoneric::AlgebraParser parser;

    {
        geoneric::String xml(parser.parse_string(geoneric::String("\"five\"")));
        BOOST_CHECK_EQUAL(xml, geoneric::String(
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
        geoneric::String xml(parser.parse_string(geoneric::String("\"\"")));
        BOOST_CHECK_EQUAL(xml, geoneric::String(
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
        geoneric::String xml(parser.parse_string(geoneric::String("\" \"")));
        BOOST_CHECK_EQUAL(xml, geoneric::String(
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
        geoneric::String xml(parser.parse_string(geoneric::String("\"mañana\"")));
        BOOST_CHECK_EQUAL(xml, geoneric::String(
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
        BOOST_CHECK_THROW(parser.parse_string(geoneric::String("if")),
            geoneric::detail::ParseError);
    }

    {
        BOOST_CHECK_THROW(parser.parse_string(geoneric::String("yield")),
            geoneric::detail::UnsupportedLanguageConstruct);
    }

    {
        BOOST_CHECK_THROW(parser.parse_string(geoneric::String("print(5)")),
            geoneric::detail::UnsupportedLanguageConstruct);
    }
}


BOOST_AUTO_TEST_CASE(parse_number)
{
    geoneric::AlgebraParser parser;

    {
        geoneric::String xml(parser.parse_string(geoneric::String("5")));
        BOOST_CHECK_EQUAL(xml, geoneric::String(boost::format(
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
        geoneric::String xml(parser.parse_string(geoneric::String("5L")));
        BOOST_CHECK_EQUAL(xml, geoneric::String(
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
        geoneric::String xml(parser.parse_string(geoneric::String("5.5")));
        BOOST_CHECK_EQUAL(xml, geoneric::String(
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
    geoneric::AlgebraParser parser;
    geoneric::String xml;

    {
        xml = parser.parse_string(geoneric::String("f()"));
        BOOST_CHECK_EQUAL(xml, geoneric::String(
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
        xml = parser.parse_string(geoneric::String("f(1, \"2\", three, four())"));
        BOOST_CHECK_EQUAL(xml, geoneric::String(boost::format(
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
//   geoneric::AlgebraParser parser;
//   geoneric::String xml;
// 
//   {
//     xml = parser.parse_string(geoneric::String("print"));
//     BOOST_CHECK_EQUAL(xml, geoneric::String(
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
//   //   xml = parser.parse_string(geoneric::String("print(1, \"2\", three, four())"));
//   //   BOOST_CHECK_EQUAL(xml, geoneric::String(boost::format(
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
    geoneric::AlgebraParser parser;
    geoneric::String xml;

    {
        xml = parser.parse_string(geoneric::String("-a"));
        BOOST_CHECK_EQUAL(xml, geoneric::String(
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
    geoneric::AlgebraParser parser;
    geoneric::String xml;

    {
        xml = parser.parse_string(geoneric::String("a + b"));
        BOOST_CHECK_EQUAL(xml, geoneric::String(
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
    geoneric::AlgebraParser parser;
    geoneric::String xml;

    {
        xml = parser.parse_string(geoneric::String("a and b"));
        BOOST_CHECK_EQUAL(xml, geoneric::String(
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
    geoneric::AlgebraParser parser;
    geoneric::String xml;

    {
        xml = parser.parse_string(geoneric::String("a <= b"));
        BOOST_CHECK_EQUAL(xml, geoneric::String(
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
    geoneric::AlgebraParser parser;

    {
        geoneric::String xml(parser.parse_string(geoneric::String("a\nb")));
        BOOST_CHECK_EQUAL(xml, geoneric::String(
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
    geoneric::AlgebraParser parser;
    geoneric::String xml;

    {
        xml = parser.parse_string(geoneric::String(
            "if a:\n"
            "  b"));
        BOOST_CHECK_EQUAL(xml, geoneric::String(
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
        xml = parser.parse_string(geoneric::String(
            "if a:\n"
            "  b\n"
            "elif(c):\n"
            "  d"));
        BOOST_CHECK_EQUAL(xml, geoneric::String(
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
        xml = parser.parse_string(geoneric::String(
            "if a:\n"
            "  b\n"
            "elif c:\n"
            "  d\n"
            "else:\n"
            "  e"));
        BOOST_CHECK_EQUAL(xml, geoneric::String(
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
    geoneric::AlgebraParser parser;
    geoneric::String xml;

    {
        xml = parser.parse_string(geoneric::String(
            "while a:\n"
            "  b"));
        BOOST_CHECK_EQUAL(xml, geoneric::String(
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
        xml = parser.parse_string(geoneric::String(
            "while a:\n"
            "  b\n"
            "else:\n"
            "  c"));
        BOOST_CHECK_EQUAL(xml, geoneric::String(
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
    geoneric::AlgebraParser parser;
    geoneric::String filename;

    {
        filename = "DoesNotExist.ran";
        BOOST_CHECK_THROW(parser.parse_file(filename),
            geoneric::detail::FileOpenError);
    }
}


BOOST_AUTO_TEST_CASE(parse_slice)
{
    geoneric::AlgebraParser parser;
    geoneric::String xml;

    {
        xml = parser.parse_string(geoneric::String(
            "a[b]"));
        BOOST_CHECK_EQUAL(xml, geoneric::String(
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
    geoneric::AlgebraParser parser;
    geoneric::String xml;

    {
        xml = parser.parse_string(geoneric::String(
            "a.b"));
        BOOST_CHECK_EQUAL(xml, geoneric::String(
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
    geoneric::AlgebraParser parser;
    geoneric::String xml;

    {
        xml = parser.parse_string(geoneric::String(
            "return"));
        BOOST_CHECK_EQUAL(xml, geoneric::String(
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
        xml = parser.parse_string(geoneric::String(
            "return c"));
        BOOST_CHECK_EQUAL(xml, geoneric::String(
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
    geoneric::AlgebraParser parser;
    geoneric::String xml;

    {
        xml = parser.parse_string(geoneric::String(
            "def foo(a, b):\n"
            "    c = a + b\n"));
        BOOST_CHECK_EQUAL(xml, geoneric::String(
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
