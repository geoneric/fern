#define BOOST_TEST_MODULE ranally script
#include <boost/test/unit_test.hpp>
#include "ranally/core/exception.h"
#include "ranally/core/string.h"
#include "ranally/script/algebra_parser.h"


BOOST_AUTO_TEST_SUITE(algebra_parser)

BOOST_AUTO_TEST_CASE(parse_empty_script)
{
    ranally::AlgebraParser parser;

    {
        ranally::String xml(parser.parse_string(ranally::String("")));
        BOOST_CHECK_EQUAL(xml, ranally::String(
            "<?xml version=\"1.0\"?>"
            "<Ranally source=\"&lt;string&gt;\">"
                "<Statements/>"
            "</Ranally>"));
    }
}


BOOST_AUTO_TEST_CASE(parse_name)
{
    ranally::AlgebraParser parser;

    {
        ranally::String xml(parser.parse_string(ranally::String("a")));
        BOOST_CHECK_EQUAL(xml, ranally::String(
          "<?xml version=\"1.0\"?>"
          "<Ranally source=\"&lt;string&gt;\">"
            "<Statements>"
              "<Statement line=\"1\" col=\"0\">"
                "<Expression line=\"1\" col=\"0\">"
                  "<Name>a</Name>"
                "</Expression>"
              "</Statement>"
            "</Statements>"
          "</Ranally>"));
    }

    // TODO #if PYTHONVER >= 2.7/3.0?
    // {
    //     ranally::String xml(parser.parse_string(ranally::String("ñ")));
    //     BOOST_CHECK_EQUAL(xml, ranally::String(
    //       "<?xml version=\"1.0\"?>"
    //       "<Ranally>"
    //         "<Statements>"
    //           "<Statement>"
    //             "<Expression line=\"1\" col=\"0\">"
    //               "<Name>ñ</Name>"
    //             "</Expression>"
    //           "</Statement>"
    //         "</Statements>"
    //       "</Ranally>"));
    // }
}


BOOST_AUTO_TEST_CASE(parse_assignment)
{
    ranally::AlgebraParser parser;

    {
        ranally::String xml(parser.parse_string(ranally::String("a = b")));
        BOOST_CHECK_EQUAL(xml, ranally::String(
            "<?xml version=\"1.0\"?>"
            "<Ranally source=\"&lt;string&gt;\">"
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
            "</Ranally>"));
    }
}


BOOST_AUTO_TEST_CASE(parse_string)
{
    ranally::AlgebraParser parser;

    {
        ranally::String xml(parser.parse_string(ranally::String("\"five\"")));
        BOOST_CHECK_EQUAL(xml, ranally::String(
            "<?xml version=\"1.0\"?>"
            "<Ranally source=\"&lt;string&gt;\">"
              "<Statements>"
                "<Statement line=\"1\" col=\"0\">"
                  "<Expression line=\"1\" col=\"0\">"
                    "<String>five</String>"
                  "</Expression>"
                "</Statement>"
              "</Statements>"
            "</Ranally>"));
    }

    {
        ranally::String xml(parser.parse_string(ranally::String("\"\"")));
        BOOST_CHECK_EQUAL(xml, ranally::String(
            "<?xml version=\"1.0\"?>"
            "<Ranally source=\"&lt;string&gt;\">"
              "<Statements>"
                "<Statement line=\"1\" col=\"0\">"
                  "<Expression line=\"1\" col=\"0\">"
                    "<String/>"
                  "</Expression>"
                "</Statement>"
              "</Statements>"
            "</Ranally>"));
    }

    {
        ranally::String xml(parser.parse_string(ranally::String("\" \"")));
        BOOST_CHECK_EQUAL(xml, ranally::String(
            "<?xml version=\"1.0\"?>"
            "<Ranally source=\"&lt;string&gt;\">"
              "<Statements>"
                "<Statement line=\"1\" col=\"0\">"
                  "<Expression line=\"1\" col=\"0\">"
                    "<String> </String>"
                  "</Expression>"
                "</Statement>"
              "</Statements>"
            "</Ranally>"));
    }

    // Test handling of Unicode characters.
    {
        ranally::String xml(parser.parse_string(ranally::String("\"mañana\"")));
        BOOST_CHECK_EQUAL(xml, ranally::String(
            "<?xml version=\"1.0\"?>"
            "<Ranally source=\"&lt;string&gt;\">"
              "<Statements>"
                "<Statement line=\"1\" col=\"0\">"
                  "<Expression line=\"1\" col=\"0\">"
                    "<String>mañana</String>"
                  "</Expression>"
                "</Statement>"
              "</Statements>"
            "</Ranally>"));
    }

    {
        BOOST_CHECK_THROW(parser.parse_string(ranally::String("if")),
            ranally::detail::ParseError);
    }

    {
        BOOST_CHECK_THROW(parser.parse_string(ranally::String("yield")),
            ranally::detail::UnsupportedLanguageConstruct);
    }

    {
        BOOST_CHECK_THROW(parser.parse_string(ranally::String("print(5)")),
            ranally::detail::UnsupportedLanguageConstruct);
    }
}


BOOST_AUTO_TEST_CASE(parse_number)
{
    ranally::AlgebraParser parser;

    {
        ranally::String xml(parser.parse_string(ranally::String("5")));
        BOOST_CHECK_EQUAL(xml, ranally::String(boost::format(
            "<?xml version=\"1.0\"?>"
            "<Ranally source=\"&lt;string&gt;\">"
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
            "</Ranally>") % (sizeof(long) * 8)));
    }

    {
        ranally::String xml(parser.parse_string(ranally::String("5L")));
        BOOST_CHECK_EQUAL(xml, ranally::String(
            "<?xml version=\"1.0\"?>"
            "<Ranally source=\"&lt;string&gt;\">"
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
            "</Ranally>"));
    }

    {
        ranally::String xml(parser.parse_string(ranally::String("5.5")));
        BOOST_CHECK_EQUAL(xml, ranally::String(
            "<?xml version=\"1.0\"?>"
            "<Ranally source=\"&lt;string&gt;\">"
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
            "</Ranally>"));
    }
}


BOOST_AUTO_TEST_CASE(parse_call)
{
    ranally::AlgebraParser parser;
    ranally::String xml;

    {
        xml = parser.parse_string(ranally::String("f()"));
        BOOST_CHECK_EQUAL(xml, ranally::String(
            "<?xml version=\"1.0\"?>"
            "<Ranally source=\"&lt;string&gt;\">"
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
            "</Ranally>"));
    }

    {
        xml = parser.parse_string(ranally::String("f(1, \"2\", three, four())"));
        BOOST_CHECK_EQUAL(xml, ranally::String(boost::format(
            "<?xml version=\"1.0\"?>"
            "<Ranally source=\"&lt;string&gt;\">"
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
            "</Ranally>") % (sizeof(long) * 8)));
    }
}


// BOOST_AUTO_TEST_CASE(parse_print)
// {
//   ranally::AlgebraParser parser;
//   ranally::String xml;
// 
//   {
//     xml = parser.parse_string(ranally::String("print"));
//     BOOST_CHECK_EQUAL(xml, ranally::String(
//       "<?xml version=\"1.0\"?>"
//       "<Ranally source=\"&lt;string&gt;\">"
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
//       "</Ranally>"));
//   }
// 
//   // {
//   //   xml = parser.parse_string(ranally::String("print(1, \"2\", three, four())"));
//   //   BOOST_CHECK_EQUAL(xml, ranally::String(boost::format(
//   //     "<?xml version=\"1.0\"?>"
//   //     "<Ranally source=\"&lt;string&gt;\">"
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
//   //     "</Ranally>") % (sizeof(long) * 8)));
//   // }
// }


BOOST_AUTO_TEST_CASE(parse_unary_operator)
{
    ranally::AlgebraParser parser;
    ranally::String xml;

    {
        xml = parser.parse_string(ranally::String("-a"));
        BOOST_CHECK_EQUAL(xml, ranally::String(
            "<?xml version=\"1.0\"?>"
            "<Ranally source=\"&lt;string&gt;\">"
              "<Statements>"
                "<Statement line=\"1\" col=\"0\">"
                  "<Expression line=\"1\" col=\"0\">"
                    "<Operator>"
                      "<Name>Sub</Name>"
                      "<Expressions>"
                        "<Expression line=\"1\" col=\"1\">"
                          "<Name>a</Name>"
                        "</Expression>"
                      "</Expressions>"
                    "</Operator>"
                  "</Expression>"
                "</Statement>"
              "</Statements>"
            "</Ranally>"));
    }
}


BOOST_AUTO_TEST_CASE(parse_binary_operator)
{
    ranally::AlgebraParser parser;
    ranally::String xml;

    {
        xml = parser.parse_string(ranally::String("a + b"));
        BOOST_CHECK_EQUAL(xml, ranally::String(
            "<?xml version=\"1.0\"?>"
            "<Ranally source=\"&lt;string&gt;\">"
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
            "</Ranally>"));
    }
}


BOOST_AUTO_TEST_CASE(parse_boolean_operator)
{
    ranally::AlgebraParser parser;
    ranally::String xml;

    {
        xml = parser.parse_string(ranally::String("a and b"));
        BOOST_CHECK_EQUAL(xml, ranally::String(
            "<?xml version=\"1.0\"?>"
            "<Ranally source=\"&lt;string&gt;\">"
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
            "</Ranally>"));
    }
}


BOOST_AUTO_TEST_CASE(parse_comparison_operator)
{
    ranally::AlgebraParser parser;
    ranally::String xml;

    {
        xml = parser.parse_string(ranally::String("a <= b"));
        BOOST_CHECK_EQUAL(xml, ranally::String(
            "<?xml version=\"1.0\"?>"
            "<Ranally source=\"&lt;string&gt;\">"
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
            "</Ranally>"));
    }
}


BOOST_AUTO_TEST_CASE(parse_multiple_statements)
{
    ranally::AlgebraParser parser;

    {
        ranally::String xml(parser.parse_string(ranally::String("a\nb")));
        BOOST_CHECK_EQUAL(xml, ranally::String(
            "<?xml version=\"1.0\"?>"
            "<Ranally source=\"&lt;string&gt;\">"
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
            "</Ranally>"));
    }
}


BOOST_AUTO_TEST_CASE(parse_if)
{
    ranally::AlgebraParser parser;
    ranally::String xml;

    {
        xml = parser.parse_string(ranally::String(
            "if a:\n"
            "  b"));
        BOOST_CHECK_EQUAL(xml, ranally::String(
            "<?xml version=\"1.0\"?>"
            "<Ranally source=\"&lt;string&gt;\">"
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
            "</Ranally>"));
    }

    {
        xml = parser.parse_string(ranally::String(
            "if a:\n"
            "  b\n"
            "elif(c):\n"
            "  d"));
        BOOST_CHECK_EQUAL(xml, ranally::String(
            "<?xml version=\"1.0\"?>"
            "<Ranally source=\"&lt;string&gt;\">"
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
            "</Ranally>"));
    }

    {
        xml = parser.parse_string(ranally::String(
            "if a:\n"
            "  b\n"
            "elif c:\n"
            "  d\n"
            "else:\n"
            "  e"));
        BOOST_CHECK_EQUAL(xml, ranally::String(
            "<?xml version=\"1.0\"?>"
            "<Ranally source=\"&lt;string&gt;\">"
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
            "</Ranally>"));
    }
}


BOOST_AUTO_TEST_CASE(parse_while)
{
    ranally::AlgebraParser parser;
    ranally::String xml;

    {
        xml = parser.parse_string(ranally::String(
            "while a:\n"
            "  b"));
        BOOST_CHECK_EQUAL(xml, ranally::String(
            "<?xml version=\"1.0\"?>"
            "<Ranally source=\"&lt;string&gt;\">"
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
            "</Ranally>"));
    }

    {
        xml = parser.parse_string(ranally::String(
            "while a:\n"
            "  b\n"
            "else:\n"
            "  c"));
        BOOST_CHECK_EQUAL(xml, ranally::String(
            "<?xml version=\"1.0\"?>"
            "<Ranally source=\"&lt;string&gt;\">"
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
            "</Ranally>"));
    }
}


BOOST_AUTO_TEST_CASE(parse_file)
{
    ranally::AlgebraParser parser;
    ranally::String filename;

    {
        filename = "DoesNotExist.ran";
        BOOST_CHECK_THROW(parser.parse_file(filename),
            ranally::detail::FileOpenError);
    }
}


BOOST_AUTO_TEST_CASE(parse_slice)
{
    ranally::AlgebraParser parser;
    ranally::String xml;

    {
        xml = parser.parse_string(ranally::String(
            "a[b]"));
        BOOST_CHECK_EQUAL(xml, ranally::String(
            "<?xml version=\"1.0\"?>"
            "<Ranally source=\"&lt;string&gt;\">"
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
            "</Ranally>"));
    }
}


BOOST_AUTO_TEST_CASE(return_)
{
    ranally::AlgebraParser parser;
    ranally::String xml;

    {
        xml = parser.parse_string(ranally::String(
            "return"));
        BOOST_CHECK_EQUAL(xml, ranally::String(
            "<?xml version=\"1.0\"?>"
            "<Ranally source=\"&lt;string&gt;\">"
              "<Statements>"
                "<Statement line=\"1\" col=\"0\">"
                  "<Return/>"
                "</Statement>"
              "</Statements>"
            "</Ranally>"));
    }

    {
        xml = parser.parse_string(ranally::String(
            "return c"));
        BOOST_CHECK_EQUAL(xml, ranally::String(
            "<?xml version=\"1.0\"?>"
            "<Ranally source=\"&lt;string&gt;\">"
              "<Statements>"
                "<Statement line=\"1\" col=\"0\">"
                  "<Return>"
                    "<Expression line=\"1\" col=\"7\">"
                      "<Name>c</Name>"
                    "</Expression>"
                  "</Return>"
                "</Statement>"
              "</Statements>"
            "</Ranally>"));
    }
}


BOOST_AUTO_TEST_CASE(parse_function)
{
    ranally::AlgebraParser parser;
    ranally::String xml;

    {
        xml = parser.parse_string(ranally::String(
            "def foo(a, b):\n"
            "    c = a + b\n"));
        BOOST_CHECK_EQUAL(xml, ranally::String(
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
            "</Ranally>"));
    }
}

BOOST_AUTO_TEST_SUITE_END()
