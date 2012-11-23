#define BOOST_TEST_MODULE ranally language
#include <boost/test/included/unit_test.hpp>
#include "Ranally/Language/algebra_parser.h"
#include "Ranally/Language/script_vertex.h"
#include "Ranally/Language/visitor.h"
#include "Ranally/Language/xml_parser.h"


class CountVerticesVisitor:
    public ranally::Visitor
{

public:

    CountVerticesVisitor()
        : ranally::Visitor(),
          _nr_vertices(0u)
    {
    }

    size_t nr_vertices() const
    {
        return _nr_vertices;
    }

private:

    size_t           _nr_vertices;

    void Visit(
        ranally::ScriptVertex& vertex)
    {
        _nr_vertices = 0u;
        ranally::Visitor::Visit(vertex);
    }

    void Visit(
        ranally::SyntaxVertex& /* vertex */)
    {
        ++_nr_vertices;
    }

};


class Support
{

public:

    Support()
        : _algebraParser(),
          _xmlParser()
    {
    }

protected:

    ranally::AlgebraParser _algebraParser;

    ranally::XmlParser _xmlParser;

};


BOOST_FIXTURE_TEST_SUITE(visitor, Support)


BOOST_AUTO_TEST_CASE(count_vertices_visitor)
{
    CountVerticesVisitor visitor;
    std::shared_ptr<ranally::ScriptVertex> tree;

    // Empty script.
    {
        // Only script vertex.
        tree = _xmlParser.parse(_algebraParser.parseString(ranally::String(
            "")));
        assert(tree);
        tree->Accept(visitor);
        BOOST_CHECK_EQUAL(visitor.nr_vertices(), 1u);
    }

    // Name.
    {
        tree = _xmlParser.parse(_algebraParser.parseString(ranally::String(
            "a")));
        assert(tree);
        tree->Accept(visitor);
        BOOST_CHECK_EQUAL(visitor.nr_vertices(), 2u);
    }

    // Number.
    {
        tree = _xmlParser.parse(_algebraParser.parseString(ranally::String(
            "5")));
        assert(tree);
        tree->Accept(visitor);
        BOOST_CHECK_EQUAL(visitor.nr_vertices(), 2u);
    }

    // String.
    {
        tree = _xmlParser.parse(_algebraParser.parseString(ranally::String(
            "\"five\"")));
        assert(tree);
        tree->Accept(visitor);
        BOOST_CHECK_EQUAL(visitor.nr_vertices(), 2u);
    }

    // Operator.
    {
        tree = _xmlParser.parse(_algebraParser.parseString(ranally::String(
            "a + b")));
        assert(tree);
        tree->Accept(visitor);
        BOOST_CHECK_EQUAL(visitor.nr_vertices(), 4u);
    }

    // Function.
    {
        tree = _xmlParser.parse(_algebraParser.parseString(ranally::String(
            "f(a, b)")));
        assert(tree);
        tree->Accept(visitor);
        BOOST_CHECK_EQUAL(visitor.nr_vertices(), 4u);
    }

    // Assignment.
    {
        tree = _xmlParser.parse(_algebraParser.parseString(ranally::String(
            "c = f(a, b)")));
        assert(tree);
        tree->Accept(visitor);
        BOOST_CHECK_EQUAL(visitor.nr_vertices(), 6u);
    }

    // If.
    {
        tree = _xmlParser.parse(_algebraParser.parseString(ranally::String(
            "if a > b:\n"
            "    c = d\n"
            "else:\n"
            "    e = f")));
        assert(tree);
        tree->Accept(visitor);
        BOOST_CHECK_EQUAL(visitor.nr_vertices(), 11u);
    }

    // While.
    {
        tree = _xmlParser.parse(_algebraParser.parseString(ranally::String(
            "while a > b:\n"
            "    c = c + d\n"
            "else:\n"
            "    e = f")));
        assert(tree);
        tree->Accept(visitor);
        BOOST_CHECK_EQUAL(visitor.nr_vertices(), 13u);
    }
}

BOOST_AUTO_TEST_SUITE_END()
