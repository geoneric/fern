#define BOOST_TEST_MODULE ranally language
#include <boost/test/included/unit_test.hpp>
#include "ranally/operation/operation-xml.h"
#include "ranally/operation/parameter.h"
#include "ranally/operation/result.h"
#include "ranally/operation/operation_xml_parser.h"
#include "ranally/language/algebra_parser.h"
#include "ranally/language/annotate_visitor.h"
#include "ranally/language/operation_vertex.h"
#include "ranally/language/script_vertex.h"
#include "ranally/language/xml_parser.h"


class Support
{

public:

    Support()
        : _algebraParser(),
          _xmlParser(),
          _visitor(ranally::OperationXmlParser().parse(ranally::operationsXml))
    {
    }

protected:

    ranally::AlgebraParser _algebraParser;

    ranally::XmlParser _xmlParser;

    ranally::AnnotateVisitor _visitor;
};


BOOST_FIXTURE_TEST_SUITE(annotate_visitor, Support)

BOOST_AUTO_TEST_CASE(visit_empty_script)
{
    // Parse empty script.
    // Ast before and after should be the same.
    std::shared_ptr<ranally::ScriptVertex> tree1, tree2;

    tree1 = _xmlParser.parse(_algebraParser.parseString(ranally::String("")));
    assert(tree1);

    // // Create copy of this empty tree.
    // ranally::CopyVisitor copyVisitor;
    // tree1->Accept(copyVisitor);
    // tree2 = copyVisitor.scriptVertex();

    tree1->Accept(_visitor);

    // // Both trees should be equal.
    // BOOST_CHECK(*tree1 == *tree2);

    BOOST_CHECK_EQUAL(tree1->sourceName(), ranally::String("<string>"));
    BOOST_CHECK_EQUAL(tree1->line(), 0);
    BOOST_CHECK_EQUAL(tree1->col(), 0);
    BOOST_CHECK(tree1->statements().empty());
}


BOOST_AUTO_TEST_CASE(visit_number)
{
    std::shared_ptr<ranally::ScriptVertex> tree =
        _xmlParser.parse(_algebraParser.parseString(ranally::String("5")));
    tree->Accept(_visitor);

    BOOST_CHECK_EQUAL(tree->sourceName(), ranally::String("<string>"));
    BOOST_CHECK_EQUAL(tree->line(), 0);
    BOOST_CHECK_EQUAL(tree->col(), 0);
    BOOST_CHECK_EQUAL(tree->statements().size(), 1u);
}


BOOST_AUTO_TEST_CASE(visit_operation)
{
    {
        std::shared_ptr<ranally::ScriptVertex> tree =
            _xmlParser.parse(_algebraParser.parseString(
                ranally::String("abs(a)")));
        tree->Accept(_visitor);

        BOOST_CHECK_EQUAL(tree->sourceName(), ranally::String("<string>"));
        BOOST_CHECK_EQUAL(tree->line(), 0);
        BOOST_CHECK_EQUAL(tree->col(), 0);

        BOOST_REQUIRE_EQUAL(tree->statements().size(), 1u);
        std::shared_ptr<ranally::StatementVertex> const& statement(
            tree->statements()[0]);
        BOOST_REQUIRE(statement);
        ranally::OperationVertex const* functionVertex(
            dynamic_cast<ranally::OperationVertex*>(statement.get()));
        BOOST_REQUIRE(functionVertex);

        ranally::OperationPtr const& operation(functionVertex->operation());
        BOOST_REQUIRE(operation);

        BOOST_CHECK_EQUAL(operation->parameters().size(), 1u);
        std::vector<ranally::Parameter> const& parameters(
            operation->parameters());
        ranally::Parameter const& parameter(parameters[0]);
        BOOST_CHECK_EQUAL(parameter.dataTypes(),
            ranally::DataTypes(ranally::DT_VALUE | ranally::DT_RASTER));
        BOOST_CHECK_EQUAL(parameter.valueTypes(),
            ranally::ValueTypes(ranally::ValueType(ranally::VT_NUMBER)));

        BOOST_CHECK_EQUAL(operation->results().size(), 1u);
        std::vector<ranally::Result> const& results(operation->results());
        ranally::Result const& result(results[0]);
        BOOST_CHECK_EQUAL(result.dataType(), ranally::DT_DEPENDS_ON_INPUT);
        BOOST_CHECK_EQUAL(result.valueType(), ranally::VT_DEPENDS_ON_INPUT);
    }
}

BOOST_AUTO_TEST_SUITE_END()
