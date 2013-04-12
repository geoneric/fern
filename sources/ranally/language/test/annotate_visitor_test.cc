#define BOOST_TEST_MODULE ranally language
#include <boost/test/unit_test.hpp>
#include "ranally/operation/core/parameter.h"
#include "ranally/operation/core/result.h"
#include "ranally/operation/xml/operation_xml_parser.h"
#include "ranally/operation/std/operation-xml.h"
#include "ranally/script/algebra_parser.h"
#include "ranally/language/annotate_visitor.h"
#include "ranally/language/vertices.h"
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

    ~Support()
    {
        _visitor.clear_stack();
    }

protected:

    ranally::AlgebraParser _algebra_parser;

    ranally::XmlParser _xml_parser;

    ranally::AnnotateVisitor _visitor;
};


// BOOST_FIXTURE_TEST_SUITE(annotate_visitor, Support)
BOOST_AUTO_TEST_SUITE(annotate_visitor)

BOOST_FIXTURE_TEST_CASE(visit_empty_script, Support)
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


BOOST_FIXTURE_TEST_CASE(visit_number, Support)
{
    {
        std::shared_ptr<ranally::ScriptVertex> tree =
            _xml_parser.parse_string(_algebra_parser.parse_string(
                ranally::String("5")));
        tree->Accept(_visitor);

        BOOST_CHECK_EQUAL(tree->source_name(), ranally::String("<string>"));
        BOOST_CHECK_EQUAL(tree->line(), 0);
        BOOST_CHECK_EQUAL(tree->col(), 0);
        BOOST_CHECK_EQUAL(tree->statements().size(), 1u);

        std::shared_ptr<ranally::StatementVertex> const& statement(
            tree->statements()[0]);
        BOOST_REQUIRE(statement);

        ranally::NumberVertex<int64_t> const* number_vertex(
            dynamic_cast<ranally::NumberVertex<int64_t>*>(statement.get()));
        BOOST_REQUIRE(number_vertex);

        ranally::ResultTypes result_types(number_vertex->result_types());
        BOOST_REQUIRE_EQUAL(result_types.size(), 1u);
        BOOST_CHECK_EQUAL(result_types[0], ranally::ResultType(
            ranally::DataTypes::SCALAR, ranally::ValueTypes::INT64));
    }

    {
        std::shared_ptr<ranally::ScriptVertex> tree =
            _xml_parser.parse_string(_algebra_parser.parse_string(
                ranally::String("5.5")));
        tree->Accept(_visitor);

        BOOST_CHECK_EQUAL(tree->source_name(), ranally::String("<string>"));
        BOOST_CHECK_EQUAL(tree->line(), 0);
        BOOST_CHECK_EQUAL(tree->col(), 0);
        BOOST_CHECK_EQUAL(tree->statements().size(), 1u);

        std::shared_ptr<ranally::StatementVertex> const& statement(
            tree->statements()[0]);
        BOOST_REQUIRE(statement);

        ranally::NumberVertex<double> const* number_vertex(
            dynamic_cast<ranally::NumberVertex<double>*>(statement.get()));
        BOOST_REQUIRE(number_vertex);

        ranally::ResultTypes result_types(number_vertex->result_types());
        BOOST_REQUIRE_EQUAL(result_types.size(), 1u);
        BOOST_CHECK_EQUAL(result_types[0], ranally::ResultType(
            ranally::DataTypes::SCALAR, ranally::ValueTypes::FLOAT64));
    }
}


BOOST_FIXTURE_TEST_CASE(visit_name, Support)
{
    std::shared_ptr<ranally::ScriptVertex> tree =
        _xml_parser.parse_string(_algebra_parser.parse_string(
            ranally::String("a")));
    tree->Accept(_visitor);

    BOOST_CHECK_EQUAL(tree->source_name(), ranally::String("<string>"));
    BOOST_CHECK_EQUAL(tree->line(), 0);
    BOOST_CHECK_EQUAL(tree->col(), 0);
    BOOST_CHECK_EQUAL(tree->statements().size(), 1u);

    std::shared_ptr<ranally::StatementVertex> const& statement(
        tree->statements()[0]);
    BOOST_REQUIRE(statement);

    ranally::NameVertex const* name_vertex(
        dynamic_cast<ranally::NameVertex*>(statement.get()));
    BOOST_REQUIRE(name_vertex);

    ranally::ResultTypes result_types(name_vertex->result_types());
    BOOST_REQUIRE_EQUAL(result_types.size(), 1u);
    BOOST_CHECK_EQUAL(result_types[0], ranally::ResultType());
}


BOOST_FIXTURE_TEST_CASE(visit_operation, Support)
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
            ranally::DataTypes::SCALAR | ranally::DataTypes::FEATURE);
        BOOST_CHECK_EQUAL(parameter.value_types(), ranally::ValueTypes::NUMBER);

        BOOST_CHECK_EQUAL(operation->results().size(), 1u);
        std::vector<ranally::Result> const& results(operation->results());
        ranally::Result const& result(results[0]);
        BOOST_CHECK_EQUAL(result.data_type(), ranally::DataTypes::ALL);
        BOOST_CHECK_EQUAL(result.value_type(), ranally::ValueTypes::NUMBER);

        ranally::ResultTypes result_types(function_vertex->result_types());
        BOOST_REQUIRE_EQUAL(result_types.size(), 1u);
        BOOST_CHECK_EQUAL(result_types[0], ranally::ResultType(
            ranally::DataTypes::ALL, ranally::ValueTypes::NUMBER));


        // abs(5), abs(-5), abs(5.5)
    }
}


class OperationResultTypeTester
{

public:

    OperationResultTypeTester(
        ranally::AlgebraParser const& algebra_parser,
        ranally::XmlParser const& xml_parser,
        ranally::AnnotateVisitor& visitor)

        : _algebra_parser(algebra_parser),
          _xml_parser(xml_parser),
          _visitor(visitor)

    {
    }

    void operator()(
        ranally::String const& script,
        ranally::ResultType const& result_type)
    {
        std::shared_ptr<ranally::ScriptVertex> tree(_xml_parser.parse_string(
            _algebra_parser.parse_string(script)));
        tree->Accept(_visitor);
        ranally::ExpressionVertex* expression_vertex =
            dynamic_cast<ranally::ExpressionVertex*>(
                tree->statements()[0].get());

        ranally::ResultTypes result_types(expression_vertex->result_types());
        BOOST_REQUIRE_EQUAL(result_types.size(), 1u);
        BOOST_CHECK_EQUAL(result_types[0], result_type);
    }

private:

    ranally::AlgebraParser const& _algebra_parser;

    ranally::XmlParser const& _xml_parser;

    ranally::AnnotateVisitor& _visitor;

};


BOOST_FIXTURE_TEST_CASE(visit_operation_2, Support)
{
    OperationResultTypeTester tester(_algebra_parser, _xml_parser, _visitor);

    // TODO Update tester from operation to expression.

    // Default integer type is int64.
    tester("5", ranally::ResultType(
        ranally::DataTypes::SCALAR,
        ranally::ValueTypes::INT64));

    // Default float type is float64.
    tester("5.5", ranally::ResultType(
        ranally::DataTypes::SCALAR,
        ranally::ValueTypes::FLOAT64));

    tester("abs(a)", ranally::ResultType(
        ranally::DataTypes::ALL,
        ranally::ValueTypes::NUMBER));
    tester("abs(5)", ranally::ResultType(
        ranally::DataTypes::SCALAR,
        ranally::ValueTypes::INT64));
    tester("abs(5.5)", ranally::ResultType(
        ranally::DataTypes::SCALAR,
        ranally::ValueTypes::FLOAT64));

    tester("int32(5)", ranally::ResultType(
        ranally::DataTypes::SCALAR,
        ranally::ValueTypes::INT32));
    tester("int32(5.5)", ranally::ResultType(
        ranally::DataTypes::SCALAR,
        ranally::ValueTypes::INT32));

    tester("5 + 6", ranally::ResultType(
        ranally::DataTypes::SCALAR,
        ranally::ValueTypes::INT64));
    tester("5 + int32(6)", ranally::ResultType(
        ranally::DataTypes::SCALAR,
        ranally::ValueTypes::INT64));
    tester("int32(5) + int32(6)", ranally::ResultType(
        ranally::DataTypes::SCALAR,
        ranally::ValueTypes::INT32));
}

BOOST_AUTO_TEST_SUITE_END()
