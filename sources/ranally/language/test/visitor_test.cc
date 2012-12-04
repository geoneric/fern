#define BOOST_TEST_MODULE ranally language
#include <boost/test/unit_test.hpp>
#include "ranally/language/algebra_parser.h"
#include "ranally/language/script_vertex.h"
#include "ranally/language/visitor.h"
#include "ranally/language/xml_parser.h"


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
        : _algebra_parser(),
          _xml_parser()
    {
    }

protected:

    ranally::AlgebraParser _algebra_parser;

    ranally::XmlParser _xml_parser;

};


BOOST_FIXTURE_TEST_SUITE(visitor, Support)


BOOST_AUTO_TEST_CASE(count_vertices_visitor)
{
    CountVerticesVisitor visitor;
    std::shared_ptr<ranally::ScriptVertex> tree;

    // Empty script.
    {
        // Only script vertex.
        tree = _xml_parser.parse_string(_algebra_parser.parse_string(
            ranally::String("")));
        assert(tree);
        tree->Accept(visitor);
        BOOST_CHECK_EQUAL(visitor.nr_vertices(), 1u);
    }

    // Name.
    {
        tree = _xml_parser.parse_string(_algebra_parser.parse_string(
            ranally::String("a")));
        assert(tree);
        tree->Accept(visitor);
        BOOST_CHECK_EQUAL(visitor.nr_vertices(), 2u);
    }

    // Number.
    {
        tree = _xml_parser.parse_string(_algebra_parser.parse_string(
            ranally::String("5")));
        assert(tree);
        tree->Accept(visitor);
        BOOST_CHECK_EQUAL(visitor.nr_vertices(), 2u);
    }

    // String.
    {
        tree = _xml_parser.parse_string(_algebra_parser.parse_string(
            ranally::String("\"five\"")));
        assert(tree);
        tree->Accept(visitor);
        BOOST_CHECK_EQUAL(visitor.nr_vertices(), 2u);
    }

    // Operator.
    {
        tree = _xml_parser.parse_string(_algebra_parser.parse_string(
            ranally::String("a + b")));
        assert(tree);
        tree->Accept(visitor);
        BOOST_CHECK_EQUAL(visitor.nr_vertices(), 4u);
    }

    // Function.
    {
        tree = _xml_parser.parse_string(_algebra_parser.parse_string(
            ranally::String("f(a, b)")));
        assert(tree);
        tree->Accept(visitor);
        BOOST_CHECK_EQUAL(visitor.nr_vertices(), 4u);
    }

    // Assignment.
    {
        tree = _xml_parser.parse_string(_algebra_parser.parse_string(
            ranally::String("c = f(a, b)")));
        assert(tree);
        tree->Accept(visitor);
        BOOST_CHECK_EQUAL(visitor.nr_vertices(), 6u);
    }

    // If.
    {
        tree = _xml_parser.parse_string(_algebra_parser.parse_string(
            ranally::String(
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
        tree = _xml_parser.parse_string(_algebra_parser.parse_string(
            ranally::String(
                "while a > b:\n"
                "    c = c + d\n"
                "else:\n"
                "    e = f")));
        assert(tree);
        tree->Accept(visitor);
        BOOST_CHECK_EQUAL(visitor.nr_vertices(), 13u);
    }

    // Slice.
    {
        tree = _xml_parser.parse_string(_algebra_parser.parse_string(
            ranally::String("a[b]")));
        assert(tree);
        tree->Accept(visitor);
        // TODO
        BOOST_CHECK_EQUAL(visitor.nr_vertices(), 13u);
    }
}

BOOST_AUTO_TEST_SUITE_END()
