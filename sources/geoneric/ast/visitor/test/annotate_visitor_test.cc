#define BOOST_TEST_MODULE geoneric ast
#include <boost/test/unit_test.hpp>
#include "geoneric/operation/core/parameter.h"
#include "geoneric/operation/core/result.h"
#include "geoneric/operation/std/operations.h"
#include "geoneric/script/algebra_parser.h"
#include "geoneric/ast/visitor/annotate_visitor.h"
#include "geoneric/ast/core/vertices.h"
#include "geoneric/ast/xml/xml_parser.h"


class Support
{

public:

    Support()
        : _algebra_parser(),
          _xml_parser(),
          _visitor(geoneric::operations())
    {
    }

    ~Support()
    {
        _visitor.clear_stack();
    }

protected:

    geoneric::AlgebraParser _algebra_parser;

    geoneric::XmlParser _xml_parser;

    geoneric::AnnotateVisitor _visitor;
};


// BOOST_FIXTURE_TEST_SUITE(annotate_visitor, Support)
BOOST_AUTO_TEST_SUITE(annotate_visitor)

BOOST_FIXTURE_TEST_CASE(visit_empty_script, Support)
{
    // Parse empty script.
    // Ast before and after should be the same.
    std::shared_ptr<geoneric::ModuleVertex> tree1, tree2;

    tree1 = _xml_parser.parse_string(_algebra_parser.parse_string(
        geoneric::String("")));
    assert(tree1);

    // // Create copy of this empty tree.
    // geoneric::CopyVisitor copyVisitor;
    // tree1->Accept(copyVisitor);
    // tree2 = copyVisitor.scriptVertex();

    tree1->Accept(_visitor);

    // // Both trees should be equal.
    // BOOST_CHECK(*tree1 == *tree2);

    BOOST_CHECK_EQUAL(tree1->source_name(), geoneric::String("<string>"));
    BOOST_CHECK_EQUAL(tree1->line(), 0);
    BOOST_CHECK_EQUAL(tree1->col(), 0);
    BOOST_CHECK(tree1->scope()->statements().empty());
}


BOOST_FIXTURE_TEST_CASE(visit_number, Support)
{
    {
        std::shared_ptr<geoneric::ModuleVertex> tree =
            _xml_parser.parse_string(_algebra_parser.parse_string(
                geoneric::String("5")));
        tree->Accept(_visitor);

        BOOST_CHECK_EQUAL(tree->source_name(), geoneric::String("<string>"));
        BOOST_CHECK_EQUAL(tree->line(), 0);
        BOOST_CHECK_EQUAL(tree->col(), 0);
        BOOST_CHECK_EQUAL(tree->scope()->statements().size(), 1u);

        std::shared_ptr<geoneric::StatementVertex> const& statement(
            tree->scope()->statements()[0]);
        BOOST_REQUIRE(statement);

        geoneric::NumberVertex<int64_t> const* number_vertex(
            dynamic_cast<geoneric::NumberVertex<int64_t>*>(statement.get()));
        BOOST_REQUIRE(number_vertex);

        geoneric::ExpressionTypes expression_types(
            number_vertex->expression_types());
        BOOST_REQUIRE_EQUAL(expression_types.size(), 1u);
        BOOST_CHECK_EQUAL(expression_types[0], geoneric::ExpressionType(
            geoneric::DataTypes::CONSTANT, geoneric::ValueTypes::INT64));
    }

    {
        std::shared_ptr<geoneric::ModuleVertex> tree =
            _xml_parser.parse_string(_algebra_parser.parse_string(
                geoneric::String("5.5")));
        tree->Accept(_visitor);

        BOOST_CHECK_EQUAL(tree->source_name(), geoneric::String("<string>"));
        BOOST_CHECK_EQUAL(tree->line(), 0);
        BOOST_CHECK_EQUAL(tree->col(), 0);
        BOOST_CHECK_EQUAL(tree->scope()->statements().size(), 1u);

        std::shared_ptr<geoneric::StatementVertex> const& statement(
            tree->scope()->statements()[0]);
        BOOST_REQUIRE(statement);

        geoneric::NumberVertex<double> const* number_vertex(
            dynamic_cast<geoneric::NumberVertex<double>*>(statement.get()));
        BOOST_REQUIRE(number_vertex);

        geoneric::ExpressionTypes expression_types(
            number_vertex->expression_types());
        BOOST_REQUIRE_EQUAL(expression_types.size(), 1u);
        BOOST_CHECK_EQUAL(expression_types[0], geoneric::ExpressionType(
            geoneric::DataTypes::CONSTANT, geoneric::ValueTypes::FLOAT64));
    }
}


BOOST_FIXTURE_TEST_CASE(visit_name, Support)
{
    std::shared_ptr<geoneric::ModuleVertex> tree =
        _xml_parser.parse_string(_algebra_parser.parse_string(
            geoneric::String("a")));
    tree->Accept(_visitor);

    BOOST_CHECK_EQUAL(tree->source_name(), geoneric::String("<string>"));
    BOOST_CHECK_EQUAL(tree->line(), 0);
    BOOST_CHECK_EQUAL(tree->col(), 0);
    BOOST_CHECK_EQUAL(tree->scope()->statements().size(), 1u);

    std::shared_ptr<geoneric::StatementVertex> const& statement(
        tree->scope()->statements()[0]);
    BOOST_REQUIRE(statement);

    geoneric::NameVertex const* name_vertex(
        dynamic_cast<geoneric::NameVertex*>(statement.get()));
    BOOST_REQUIRE(name_vertex);

    geoneric::ExpressionTypes expression_types(name_vertex->expression_types());
    BOOST_REQUIRE_EQUAL(expression_types.size(), 1u);
    BOOST_CHECK_EQUAL(expression_types[0], geoneric::ExpressionType());
}


BOOST_FIXTURE_TEST_CASE(visit_operation, Support)
{
    {
        std::shared_ptr<geoneric::ModuleVertex> tree =
            _xml_parser.parse_string(_algebra_parser.parse_string(
                geoneric::String("abs(a)")));
        tree->Accept(_visitor);

        BOOST_CHECK_EQUAL(tree->source_name(), geoneric::String("<string>"));
        BOOST_CHECK_EQUAL(tree->line(), 0);
        BOOST_CHECK_EQUAL(tree->col(), 0);

        BOOST_REQUIRE_EQUAL(tree->scope()->statements().size(), 1u);
        std::shared_ptr<geoneric::StatementVertex> const& statement(
            tree->scope()->statements()[0]);
        BOOST_REQUIRE(statement);
        geoneric::OperationVertex const* function_call_vertex(
            dynamic_cast<geoneric::OperationVertex*>(statement.get()));
        BOOST_REQUIRE(function_call_vertex);

        geoneric::OperationPtr const& operation(
            function_call_vertex->operation());
        BOOST_REQUIRE(operation);

        BOOST_CHECK_EQUAL(operation->parameters().size(), 1u);
        std::vector<geoneric::Parameter> const& parameters(
            operation->parameters());
        geoneric::Parameter const& parameter(parameters[0]);
        BOOST_CHECK_EQUAL(parameter.expression_types().size(), 1u);
        BOOST_CHECK_EQUAL(parameter.expression_types()[0].data_type(),
            geoneric::DataTypes::CONSTANT | geoneric::DataTypes::STATIC_FIELD);
        BOOST_CHECK_EQUAL(parameter.expression_types()[0].value_type(),
            geoneric::ValueTypes::NUMBER);

        BOOST_CHECK_EQUAL(operation->results().size(), 1u);
        std::vector<geoneric::Result> const& results(operation->results());
        geoneric::Result const& result(results[0]);
        BOOST_CHECK_EQUAL(result.expression_type().data_type(),
            geoneric::DataTypes::ALL);
        BOOST_CHECK_EQUAL(result.expression_type().value_type(),
            geoneric::ValueTypes::NUMBER);

        geoneric::ExpressionTypes expression_types(
            function_call_vertex->expression_types());
        BOOST_REQUIRE_EQUAL(expression_types.size(), 1u);
        BOOST_CHECK_EQUAL(expression_types[0], geoneric::ExpressionType(
            geoneric::DataTypes::ALL, geoneric::ValueTypes::NUMBER));


        // abs(5), abs(-5), abs(5.5)
    }

    // Test whether result data type of read is propagated to result type of
    // abs. Read results in field, so abs must result in field, with the same
    // value types.
    {
        std::shared_ptr<geoneric::ModuleVertex> tree =
            _xml_parser.parse_string(_algebra_parser.parse_string(
                geoneric::String("abs(read(\"a\"))")));
        tree->Accept(_visitor);

        BOOST_CHECK_EQUAL(tree->source_name(), geoneric::String("<string>"));
        BOOST_CHECK_EQUAL(tree->line(), 0);
        BOOST_CHECK_EQUAL(tree->col(), 0);

        BOOST_REQUIRE_EQUAL(tree->scope()->statements().size(), 1u);
        std::shared_ptr<geoneric::StatementVertex> const& statement(
            tree->scope()->statements()[0]);
        BOOST_REQUIRE(statement);
        geoneric::OperationVertex const* abs_vertex(
            dynamic_cast<geoneric::OperationVertex*>(statement.get()));
        BOOST_REQUIRE(abs_vertex);

        geoneric::ExpressionTypes expression_types(
            abs_vertex->expression_types());
        BOOST_REQUIRE_EQUAL(expression_types.size(), 1u);
        BOOST_CHECK_EQUAL(expression_types[0], geoneric::ExpressionType(
            geoneric::DataTypes::STATIC_FIELD, geoneric::ValueTypes::NUMBER));
    }
}


class OperationResultTypeTester
{

public:

    OperationResultTypeTester(
        geoneric::AlgebraParser const& algebra_parser,
        geoneric::XmlParser const& xml_parser,
        geoneric::AnnotateVisitor& visitor)

        : _algebra_parser(algebra_parser),
          _xml_parser(xml_parser),
          _visitor(visitor)

    {
    }

    void operator()(
        geoneric::String const& script,
        geoneric::ExpressionType const& expression_type)
    {
        std::shared_ptr<geoneric::ModuleVertex> tree(_xml_parser.parse_string(
            _algebra_parser.parse_string(script)));
        tree->Accept(_visitor);
        geoneric::ExpressionVertex* expression_vertex =
            dynamic_cast<geoneric::ExpressionVertex*>(
                tree->scope()->statements()[0].get());

        geoneric::ExpressionTypes expression_types(
            expression_vertex->expression_types());
        BOOST_REQUIRE_EQUAL(expression_types.size(), 1u);
        BOOST_CHECK_EQUAL(expression_types[0], expression_type);
    }

private:

    geoneric::AlgebraParser const& _algebra_parser;

    geoneric::XmlParser const& _xml_parser;

    geoneric::AnnotateVisitor& _visitor;

};


BOOST_FIXTURE_TEST_CASE(visit_operation_2, Support)
{
    OperationResultTypeTester tester(_algebra_parser, _xml_parser, _visitor);

    // TODO Update tester from operation to expression.

    // Default integer type is int64.
    tester("5", geoneric::ExpressionType(
        geoneric::DataTypes::CONSTANT,
        geoneric::ValueTypes::INT64));

    // Default float type is float64.
    tester("5.5", geoneric::ExpressionType(
        geoneric::DataTypes::CONSTANT,
        geoneric::ValueTypes::FLOAT64));

    tester("abs(a)", geoneric::ExpressionType(
        geoneric::DataTypes::ALL,
        geoneric::ValueTypes::NUMBER));
    tester("abs(5)", geoneric::ExpressionType(
        geoneric::DataTypes::CONSTANT,
        geoneric::ValueTypes::INT64));
    tester("abs(5.5)", geoneric::ExpressionType(
        geoneric::DataTypes::CONSTANT,
        geoneric::ValueTypes::FLOAT64));

    tester("int32(5)", geoneric::ExpressionType(
        geoneric::DataTypes::CONSTANT,
        geoneric::ValueTypes::INT32));
    tester("int32(5.5)", geoneric::ExpressionType(
        geoneric::DataTypes::CONSTANT,
        geoneric::ValueTypes::INT32));

    tester("5 + 6", geoneric::ExpressionType(
        geoneric::DataTypes::CONSTANT,
        geoneric::ValueTypes::INT64));
    tester("5 + int32(6)", geoneric::ExpressionType(
        geoneric::DataTypes::CONSTANT,
        geoneric::ValueTypes::INT64));
    tester("int32(5) + int32(6)", geoneric::ExpressionType(
        geoneric::DataTypes::CONSTANT,
        geoneric::ValueTypes::INT32));
}


BOOST_FIXTURE_TEST_CASE(visit_attribute, Support)
{
    {
        // TODO hier verder
        // std::shared_ptr<geoneric::ModuleVertex> tree =
        //     _xml_parser.parse_string(_algebra_parser.parse_string(
        //         geoneric::String("5")));
        // tree->Accept(_visitor);

        // BOOST_CHECK_EQUAL(tree->source_name(), geoneric::String("<string>"));
        // BOOST_CHECK_EQUAL(tree->line(), 0);
        // BOOST_CHECK_EQUAL(tree->col(), 0);
        // BOOST_CHECK_EQUAL(tree->scope()->statements().size(), 1u);

        // std::shared_ptr<geoneric::StatementVertex> const& statement(
        //     tree->scope()->statements()[0]);
        // BOOST_REQUIRE(statement);

        // geoneric::NumberVertex<int64_t> const* number_vertex(
        //     dynamic_cast<geoneric::NumberVertex<int64_t>*>(statement.get()));
        // BOOST_REQUIRE(number_vertex);

        // geoneric::ExpressionTypes expression_types(
        //     number_vertex->expression_types());
        // BOOST_REQUIRE_EQUAL(expression_types.size(), 1u);
        // BOOST_CHECK_EQUAL(expression_types[0], geoneric::ExpressionType(
        //     geoneric::DataTypes::CONSTANT, geoneric::ValueTypes::INT64));
    }
}

BOOST_AUTO_TEST_SUITE_END()
