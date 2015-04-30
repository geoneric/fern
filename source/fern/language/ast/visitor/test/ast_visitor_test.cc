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
#include "fern/language/script/algebra_parser.h"
#include "fern/language/ast/core/module_vertex.h"
#include "fern/language/ast/visitor/ast_visitor.h"
#include "fern/language/ast/xml/xml_parser.h"


class CountVerticesVisitor:
    public fern::AstVisitor
{

public:

    CountVerticesVisitor()
        : fern::AstVisitor(),
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
        fern::ModuleVertex& vertex)
    {
        _nr_vertices = 0u;
        fern::AstVisitor::Visit(vertex);
    }

    void Visit(
        fern::AstVertex& /* vertex */)
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

    fern::AlgebraParser _algebra_parser;

    fern::XmlParser _xml_parser;

};


BOOST_FIXTURE_TEST_SUITE(visitor, Support)


BOOST_AUTO_TEST_CASE(count_vertices_visitor)
{
    CountVerticesVisitor visitor;
    std::shared_ptr<fern::ModuleVertex> tree;

    // Empty script.
    {
        // Script, scope, sentinel.
        tree = _xml_parser.parse_string(_algebra_parser.parse_string(
            fern::String("")));
        assert(tree);
        tree->Accept(visitor);
        BOOST_CHECK_EQUAL(visitor.nr_vertices(), 3u);
    }

    // Name.
    {
        // Script, scope, name, sentinel.
        tree = _xml_parser.parse_string(_algebra_parser.parse_string(
            fern::String("a")));
        assert(tree);
        tree->Accept(visitor);
        BOOST_CHECK_EQUAL(visitor.nr_vertices(), 4u);
    }

    // Number.
    {
        tree = _xml_parser.parse_string(_algebra_parser.parse_string(
            fern::String("5")));
        assert(tree);
        tree->Accept(visitor);
        BOOST_CHECK_EQUAL(visitor.nr_vertices(), 4u);
    }

    // String.
    {
        tree = _xml_parser.parse_string(_algebra_parser.parse_string(
            fern::String("\"five\"")));
        assert(tree);
        tree->Accept(visitor);
        BOOST_CHECK_EQUAL(visitor.nr_vertices(), 4u);
    }

    // Operator.
    {
        tree = _xml_parser.parse_string(_algebra_parser.parse_string(
            fern::String("a + b")));
        assert(tree);
        tree->Accept(visitor);
        BOOST_CHECK_EQUAL(visitor.nr_vertices(), 6u);
    }

    // Function.
    {
        tree = _xml_parser.parse_string(_algebra_parser.parse_string(
            fern::String("f(a, b)")));
        assert(tree);
        tree->Accept(visitor);
        BOOST_CHECK_EQUAL(visitor.nr_vertices(), 6u);
    }

    // Assignment.
    {
        tree = _xml_parser.parse_string(_algebra_parser.parse_string(
            fern::String("c = f(a, b)")));
        assert(tree);
        tree->Accept(visitor);
        BOOST_CHECK_EQUAL(visitor.nr_vertices(), 8u);
    }

    // If.
    {
        tree = _xml_parser.parse_string(_algebra_parser.parse_string(
            fern::String(
                "if a > b:\n"
                "    c = d\n"
                "else:\n"
                "    e = f")));
        assert(tree);
        tree->Accept(visitor);
        BOOST_CHECK_EQUAL(visitor.nr_vertices(), 17u);
    }

    // While.
    {
        tree = _xml_parser.parse_string(_algebra_parser.parse_string(
            fern::String(
                "while a > b:\n"
                "    c = c + d\n"
                "else:\n"
                "    e = f")));
        assert(tree);
        tree->Accept(visitor);
        BOOST_CHECK_EQUAL(visitor.nr_vertices(), 19u);
    }

    // Slice.
    {
        tree = _xml_parser.parse_string(_algebra_parser.parse_string(
            fern::String("a[b]")));
        assert(tree);
        tree->Accept(visitor);
        BOOST_CHECK_EQUAL(visitor.nr_vertices(), 6u);
    }

    // Attribute.
    {
        tree = _xml_parser.parse_string(_algebra_parser.parse_string(
            fern::String("a.b")));
        assert(tree);
        tree->Accept(visitor);
        BOOST_CHECK_EQUAL(visitor.nr_vertices(), 5u);
    }

    // Function definition.
    {
        tree = _xml_parser.parse_string(_algebra_parser.parse_string(
            fern::String(u8R"(
def foo():
    return
)")));
        assert(tree);
        tree->Accept(visitor);
        BOOST_CHECK_EQUAL(visitor.nr_vertices(), 7u);
    }

    // Function call.
    {
        tree = _xml_parser.parse_string(_algebra_parser.parse_string(
            fern::String(u8R"(
bla()
)")));
        assert(tree);
        tree->Accept(visitor);
        BOOST_CHECK_EQUAL(visitor.nr_vertices(), 4u);
    }
}

BOOST_AUTO_TEST_SUITE_END()
