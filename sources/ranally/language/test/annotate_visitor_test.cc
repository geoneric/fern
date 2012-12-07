#define BOOST_TEST_MODULE ranally language
#include <boost/test/unit_test.hpp>
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
        : _algebra_parser(),
          _xml_parser(),
          _visitor(ranally::OperationXmlParser().parse(
              ranally::operations_xml))
    {
    }

protected:

    ranally::AlgebraParser _algebra_parser;

    ranally::XmlParser _xml_parser;

    ranally::AnnotateVisitor _visitor;
};


BOOST_FIXTURE_TEST_SUITE(annotate_visitor, Support)

BOOST_AUTO_TEST_CASE(visit_empty_script)
{
    // Parse empty script.
    // Ast before and after should be the same.
    std::shared_ptr<ranally::ScriptVertex> tree1, tree2;

    tree1 = _xml_parser.parse_string(_algebra_parser.parse_string(
        ranally::String("")));
    assert(tree1);

    // // Create copy of this empty tree.
    // ranally::CopyVisitor copyVisitor;
    // tree1->Accept(copyVisitor);
    // tree2 = copyVisitor.scriptVertex();

    tree1->Accept(_visitor);

    // // Both trees should be equal.
    // BOOST_CHECK(*tree1 == *tree2);

    BOOST_CHECK_EQUAL(tree1->source_name(), ranally::String("<string>"));
    BOOST_CHECK_EQUAL(tree1->line(), 0);
    BOOST_CHECK_EQUAL(tree1->col(), 0);
    BOOST_CHECK(tree1->statements().empty());
}


BOOST_AUTO_TEST_CASE(visit_number)
{
    std::shared_ptr<ranally::ScriptVertex> tree =
        _xml_parser.parse_string(_algebra_parser.parse_string(
            ranally::String("5")));
    tree->Accept(_visitor);

    BOOST_CHECK_EQUAL(tree->source_name(), ranally::String("<string>"));
    BOOST_CHECK_EQUAL(tree->line(), 0);
    BOOST_CHECK_EQUAL(tree->col(), 0);
    BOOST_CHECK_EQUAL(tree->statements().size(), 1u);
}


BOOST_AUTO_TEST_CASE(visit_operation)
{
    {
        std::shared_ptr<ranally::ScriptVertex> tree =
            _xml_parser.parse_string(_algebra_parser.parse_string(
                ranally::String("abs(a)")));
        tree->Accept(_visitor);

        BOOST_CHECK_EQUAL(tree->source_name(), ranally::String("<string>"));
        BOOST_CHECK_EQUAL(tree->line(), 0);
        BOOST_CHECK_EQUAL(tree->col(), 0);

        BOOST_REQUIRE_EQUAL(tree->statements().size(), 1u);
        std::shared_ptr<ranally::StatementVertex> const& statement(
            tree->statements()[0]);
        BOOST_REQUIRE(statement);
        ranally::OperationVertex const* function_vertex(
            dynamic_cast<ranally::OperationVertex*>(statement.get()));
        BOOST_REQUIRE(function_vertex);

        ranally::OperationPtr const& operation(function_vertex->operation());
        BOOST_REQUIRE(operation);

        BOOST_CHECK_EQUAL(operation->parameters().size(), 1u);
        std::vector<ranally::Parameter> const& parameters(
            operation->parameters());
        ranally::Parameter const& parameter(parameters[0]);
        BOOST_CHECK_EQUAL(parameter.data_types(),
            ranally::DataTypes(ranally::DT_SCALAR | ranally::DT_FEATURE));
        BOOST_CHECK_EQUAL(parameter.value_types(),
            ranally::ValueTypes(ranally::ValueType(ranally::VT_NUMBER)));

        BOOST_CHECK_EQUAL(operation->results().size(), 1u);
        std::vector<ranally::Result> const& results(operation->results());
        ranally::Result const& result(results[0]);
        BOOST_CHECK_EQUAL(result.data_type(), ranally::DT_DEPENDS_ON_INPUT);
        BOOST_CHECK_EQUAL(result.value_type(), ranally::VT_DEPENDS_ON_INPUT);
    }
}

BOOST_AUTO_TEST_SUITE_END()
