#define BOOST_TEST_MODULE ranally ast
#include <boost/test/unit_test.hpp>
#include "ranally/script/algebra_parser.h"
#include "ranally/ast/core/vertices.h"
#include "ranally/ast/visitor/thread_visitor.h"
#include "ranally/ast/visitor/validate_visitor.h"
#include "ranally/ast/xml/xml_parser.h"


class Support
{

public:

    Support()
        : _algebra_parser(),
          _xml_parser(),
          _thread_visitor()
    {
    }

protected:

    ranally::AlgebraParser _algebra_parser;

    ranally::XmlParser _xml_parser;

    ranally::ThreadVisitor _thread_visitor;

    ranally::ValidateVisitor _validate_visitor;

};


BOOST_FIXTURE_TEST_SUITE(validate_visitor, Support)

BOOST_AUTO_TEST_CASE(visit_function_definition)
{
    bool test_implemented = false;
    BOOST_WARN(test_implemented);

    std::shared_ptr<ranally::ModuleVertex> tree;

    {
        tree = _xml_parser.parse_string(_algebra_parser.parse_string(
            ranally::String(u8R"(
def foo():
    return

bar()
)")));

        tree->Accept(_thread_visitor);
        // tree->Accept(_validate_visitor);
    }
}

BOOST_AUTO_TEST_SUITE_END()
