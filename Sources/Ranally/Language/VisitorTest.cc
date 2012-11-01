#include "VisitorTest.h"
#include <boost/test/test_tools.hpp>
#include <boost/test/unit_test_suite.hpp>
#include "Ranally/Language/ScriptVertex.h"
#include "Ranally/Language/Visitor.h"


namespace {

class CountVerticesVisitor:
    public ranally::Visitor
{

public:

    CountVerticesVisitor()
        : ranally::Visitor(),
          _nrVertices(0u)
    {
    }

    size_t nrVertices() const
    {
        return _nrVertices;
    }

private:

    size_t           _nrVertices;

    void Visit(
        ranally::ScriptVertex& vertex)
    {
        _nrVertices = 0u;
        ranally::Visitor::Visit(vertex);
    }

    void Visit(
        ranally::SyntaxVertex& /* vertex */)
    {
        ++_nrVertices;
    }

};

} // Anonymous namespace


boost::unit_test::test_suite* VisitorTest::suite()
{
    boost::unit_test::test_suite* suite = BOOST_TEST_SUITE(__FILE__);
    boost::shared_ptr<VisitorTest> instance(
        new VisitorTest());
    suite->add(BOOST_CLASS_TEST_CASE(
        &VisitorTest::testCountVerticesVisitor, instance));

    return suite;
}


VisitorTest::VisitorTest()
{
}


void VisitorTest::testCountVerticesVisitor()
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
        BOOST_CHECK_EQUAL(visitor.nrVertices(), 1u);
    }

    // Name.
    {
        tree = _xmlParser.parse(_algebraParser.parseString(ranally::String(
            "a")));
        assert(tree);
        tree->Accept(visitor);
        BOOST_CHECK_EQUAL(visitor.nrVertices(), 2u);
    }

    // Number.
    {
        tree = _xmlParser.parse(_algebraParser.parseString(ranally::String(
            "5")));
        assert(tree);
        tree->Accept(visitor);
        BOOST_CHECK_EQUAL(visitor.nrVertices(), 2u);
    }

    // String.
    {
        tree = _xmlParser.parse(_algebraParser.parseString(ranally::String(
            "\"five\"")));
        assert(tree);
        tree->Accept(visitor);
        BOOST_CHECK_EQUAL(visitor.nrVertices(), 2u);
    }

    // Operator.
    {
        tree = _xmlParser.parse(_algebraParser.parseString(ranally::String(
            "a + b")));
        assert(tree);
        tree->Accept(visitor);
        BOOST_CHECK_EQUAL(visitor.nrVertices(), 4u);
    }

    // Function.
    {
        tree = _xmlParser.parse(_algebraParser.parseString(ranally::String(
            "f(a, b)")));
        assert(tree);
        tree->Accept(visitor);
        BOOST_CHECK_EQUAL(visitor.nrVertices(), 4u);
    }

    // Assignment.
    {
        tree = _xmlParser.parse(_algebraParser.parseString(ranally::String(
            "c = f(a, b)")));
        assert(tree);
        tree->Accept(visitor);
        BOOST_CHECK_EQUAL(visitor.nrVertices(), 6u);
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
        BOOST_CHECK_EQUAL(visitor.nrVertices(), 11u);
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
        BOOST_CHECK_EQUAL(visitor.nrVertices(), 13u);
    }
}
