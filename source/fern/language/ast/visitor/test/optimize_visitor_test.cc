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
#include "fern/interpreter/Interpreter.h"
#include "fern/language/ast/optimize_visitor.h"
#include "fern/language/ast/script_visitor.h"


class Support
{

public:

    Support()
        : _interpreter(),
          _script_visitor(),
          _optimize_visitor()
    {
    }

protected:

    fern::Interpreter _interpreter;

    fern::ScriptVisitor _script_visitor;

    fern::OptimizeVisitor _optimize_visitor;
};


BOOST_FIXTURE_TEST_SUITE(optimize_visitor, Support)


BOOST_AUTO_TEST_CASE(remove_temporary_identifier)
{
    std::shared_ptr<fern::ModuleVertex> tree;
    std::string script;

    // Make sure that temporary identifiers which are only used as input to
    // one expression, are removed.
    {
        // This script should be rewritten in the tree as d = 5.
        script =
            "a = 5\n"
            "d = a\n";
        tree = _interpreter.parse_string(script);
        _interpreter.annotate(tree);
        tree->Accept(_optimize_visitor);
        tree->Accept(_script_visitor);
        BOOST_CHECK_EQUAL(_script_visitor.script(), "d = 5\n");

        // This script should be rewritten in the tree as e = 5.
        script =
            "a = 5\n"
            "d = a\n"
            "e = d\n";
        tree = _interpreter.parse_string(script);
        _interpreter.annotate(tree);
std::cout << "--------------------------" << std::endl;
        tree->Accept(_optimize_visitor);
        tree->Accept(_script_visitor);
std::cout << _script_visitor.script().encode_in_utf8() << std::endl;
        BOOST_CHECK_EQUAL(_script_visitor.script(), "e = 5\n");

        return;

        // This script should be rewritten in the tree as d = f(5).
        script =
            "a = 5\n"
            "d = f(a)\n";
        tree = _interpreter.parse_string(script);
        _interpreter.annotate(tree);
        tree->Accept(_optimize_visitor);
        tree->Accept(_script_visitor);
        std::cout << _script_visitor.script().encode_in_utf8() << std::endl;
        BOOST_CHECK_EQUAL(_script_visitor.script(),
            "d = f(5)");
    }

    // Make sure that temporary identifiers which are only used as input to
    // more than one expression, are removed.
    {
        // This script should not be rewritten.
        script =
            "a = 5\n"
            "d = f(a)\n"
            "e = g(a)\n";
        tree = _interpreter.parse_string(script);
        _interpreter.annotate(tree);
        tree->Accept(_optimize_visitor);
        tree->Accept(_script_visitor);
        BOOST_CHECK_EQUAL(_script_visitor.script(), script);
    }
}

BOOST_AUTO_TEST_SUITE_END()
