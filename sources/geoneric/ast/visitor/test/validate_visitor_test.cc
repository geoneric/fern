#define BOOST_TEST_MODULE geoneric ast
#include <boost/test/unit_test.hpp>
#include "geoneric/core/validate_error.h"
#include "geoneric/script/algebra_parser.h"
#include "geoneric/ast/core/vertices.h"
#include "geoneric/ast/visitor/thread_visitor.h"
#include "geoneric/ast/visitor/validate_visitor.h"
#include "geoneric/ast/xml/xml_parser.h"


class Support
{

public:

    Support()
        : _algebra_parser(),
          _xml_parser(),
          _thread_visitor(),
          _validate_visitor()
    {
    }

protected:

    geoneric::AlgebraParser _algebra_parser;

    geoneric::XmlParser _xml_parser;

    geoneric::ThreadVisitor _thread_visitor;

    geoneric::ValidateVisitor _validate_visitor;

};


BOOST_FIXTURE_TEST_SUITE(validate_visitor, Support)

BOOST_AUTO_TEST_CASE(visit_function_definition)
{
    std::shared_ptr<geoneric::ModuleVertex> tree;

    // Call undefined operation. Not built-in and not user-defined.
    {
        tree = _xml_parser.parse_string(_algebra_parser.parse_string(
            geoneric::String(u8R"(
def foo():
    return

bar()
)")));

        tree->Accept(_thread_visitor);
        BOOST_CHECK_THROW(tree->Accept(_validate_visitor),
            geoneric::detail::UndefinedOperation);
    }
}

BOOST_AUTO_TEST_SUITE_END()
