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
#include "fern/core/validate_error.h"
#include "fern/script/algebra_parser.h"
#include "fern/ast/core/vertices.h"
#include "fern/ast/visitor/thread_visitor.h"
#include "fern/ast/visitor/validate_visitor.h"
#include "fern/ast/xml/xml_parser.h"


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

    fern::AlgebraParser _algebra_parser;

    fern::XmlParser _xml_parser;

    fern::ThreadVisitor _thread_visitor;

    fern::ValidateVisitor _validate_visitor;

};


BOOST_FIXTURE_TEST_SUITE(validate_visitor, Support)

BOOST_AUTO_TEST_CASE(visit_function_definition)
{
    std::shared_ptr<fern::ModuleVertex> tree;

    // Call undefined operation. Not built-in and not user-defined.
    {
        tree = _xml_parser.parse_string(_algebra_parser.parse_string(
            fern::String(u8R"(
def foo():
    return

bar()
)")));

        tree->Accept(_thread_visitor);
        BOOST_CHECK_THROW(tree->Accept(_validate_visitor),
            fern::detail::UndefinedOperation);
    }
}

BOOST_AUTO_TEST_SUITE_END()
