#define BOOST_TEST_MODULE ranally language
#include <boost/test/included/unit_test.hpp>
#include "Ranally/Interpreter/Interpreter.h"
#include "Ranally/Language/OptimizeVisitor.h"
#include "Ranally/Language/ScriptVisitor.h"
#include "Ranally/Util/String.h"


class Support
{

public:

    Support()
        : _interpreter(),
          _scriptVisitor(),
          _optimizeVisitor()
    {
    }

protected:

    ranally::Interpreter _interpreter;

    ranally::ScriptVisitor _scriptVisitor;

    ranally::OptimizeVisitor _optimizeVisitor;
};


BOOST_FIXTURE_TEST_SUITE(optimize_visitor, Support)


BOOST_AUTO_TEST_CASE(remove_temporary_identifier)
{
    std::shared_ptr<ranally::ScriptVertex> tree;
    ranally::String script;

    // Make sure that temporary identifiers which are only used as input to
    // one expression, are removed.
    {
        // This script should be rewritten in the tree as d = 5.
        script = ranally::String(
            "a = 5\n"
            "d = a\n");
        tree = _interpreter.parseString(script);
        _interpreter.annotate(tree);
        tree->Accept(_optimizeVisitor);
        tree->Accept(_scriptVisitor);
        BOOST_CHECK_EQUAL(_scriptVisitor.script(), ranally::String("d = 5\n"));

        // This script should be rewritten in the tree as e = 5.
        script = ranally::String(
            "a = 5\n"
            "d = a\n"
            "e = d\n"
        );
        tree = _interpreter.parseString(script);
        _interpreter.annotate(tree);
std::cout << "--------------------------" << std::endl;
        tree->Accept(_optimizeVisitor);
        tree->Accept(_scriptVisitor);
std::cout << _scriptVisitor.script().encodeInUTF8() << std::endl;
        BOOST_CHECK_EQUAL(_scriptVisitor.script(), ranally::String("e = 5\n"));

        return;

        // This script should be rewritten in the tree as d = f(5).
        script = ranally::String(
            "a = 5\n"
            "d = f(a)\n");
        tree = _interpreter.parseString(script);
        _interpreter.annotate(tree);
        tree->Accept(_optimizeVisitor);
        tree->Accept(_scriptVisitor);
        std::cout << _scriptVisitor.script().encodeInUTF8() << std::endl;
        BOOST_CHECK_EQUAL(_scriptVisitor.script(), ranally::String("d = f(5)"));
    }

    // Make sure that temporary identifiers which are only used as input to
    // more than one expression, are removed.
    {
        // This script should not be rewritten.
        script = ranally::String(
            "a = 5\n"
            "d = f(a)\n"
            "e = g(a)\n");
        tree = _interpreter.parseString(script);
        _interpreter.annotate(tree);
        tree->Accept(_optimizeVisitor);
        tree->Accept(_scriptVisitor);
        BOOST_CHECK_EQUAL(_scriptVisitor.script(), script);
    }
}

BOOST_AUTO_TEST_SUITE_END()
