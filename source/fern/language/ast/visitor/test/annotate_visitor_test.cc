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
#include "fern/language/operation/core/parameter.h"
#include "fern/language/operation/core/result.h"
#include "fern/language/operation/std/operations.h"
#include "fern/language/script/algebra_parser.h"
#include "fern/language/ast/visitor/annotate_visitor.h"
#include "fern/language/ast/core/vertices.h"
#include "fern/language/ast/xml/xml_parser.h"


namespace fl = fern::language;


class Support
{

public:

    Support()
        : _algebra_parser(),
          _xml_parser(),
          _visitor(fl::operations())
    {
    }

    ~Support()
    {
        _visitor.clear_stack();
    }

protected:

    fl::AlgebraParser _algebra_parser;

    fl::XmlParser _xml_parser;

    fl::AnnotateVisitor _visitor;
};


// BOOST_FIXTURE_TEST_SUITE(annotate_visitor, Support)
BOOST_AUTO_TEST_SUITE(annotate_visitor)

BOOST_FIXTURE_TEST_CASE(visit_empty_script, Support)
{
    // Parse empty script.
    // Ast before and after should be the same.
    std::shared_ptr<fl::ModuleVertex> tree1, tree2;

    tree1 = _xml_parser.parse_string(_algebra_parser.parse_string(""));
    assert(tree1);

    // // Create copy of this empty tree.
    // fl::CopyVisitor copyVisitor;
    // tree1->Accept(copyVisitor);
    // tree2 = copyVisitor.scriptVertex();

    tree1->Accept(_visitor);

    // // Both trees should be equal.
    // BOOST_CHECK(*tree1 == *tree2);

    BOOST_CHECK_EQUAL(tree1->source_name(), "<string>");
    BOOST_CHECK_EQUAL(tree1->line(), 0);
    BOOST_CHECK_EQUAL(tree1->col(), 0);
    BOOST_CHECK(tree1->scope()->statements().empty());
}


BOOST_FIXTURE_TEST_CASE(visit_number, Support)
{
    {
        std::shared_ptr<fl::ModuleVertex> tree =
            _xml_parser.parse_string(_algebra_parser.parse_string("5"));
        tree->Accept(_visitor);

        BOOST_CHECK_EQUAL(tree->source_name(), "<string>");
        BOOST_CHECK_EQUAL(tree->line(), 0);
        BOOST_CHECK_EQUAL(tree->col(), 0);
        BOOST_CHECK_EQUAL(tree->scope()->statements().size(), 1u);

        std::shared_ptr<fl::StatementVertex> const& statement(
            tree->scope()->statements()[0]);
        BOOST_REQUIRE(statement);

        fl::NumberVertex<int64_t> const* number_vertex(
            dynamic_cast<fl::NumberVertex<int64_t>*>(statement.get()));
        BOOST_REQUIRE(number_vertex);

        fl::ExpressionTypes expression_types(
            number_vertex->expression_types());
        BOOST_REQUIRE_EQUAL(expression_types.size(), 1u);
        BOOST_CHECK_EQUAL(expression_types[0], fern::ExpressionType(
            fern::DataTypes::CONSTANT, fern::ValueTypes::INT64));
    }

    {
        std::shared_ptr<fl::ModuleVertex> tree =
            _xml_parser.parse_string(_algebra_parser.parse_string("5.5"));
        tree->Accept(_visitor);

        BOOST_CHECK_EQUAL(tree->source_name(), "<string>");
        BOOST_CHECK_EQUAL(tree->line(), 0);
        BOOST_CHECK_EQUAL(tree->col(), 0);
        BOOST_CHECK_EQUAL(tree->scope()->statements().size(), 1u);

        std::shared_ptr<fl::StatementVertex> const& statement(
            tree->scope()->statements()[0]);
        BOOST_REQUIRE(statement);

        fl::NumberVertex<double> const* number_vertex(
            dynamic_cast<fl::NumberVertex<double>*>(statement.get()));
        BOOST_REQUIRE(number_vertex);

        fl::ExpressionTypes expression_types(
            number_vertex->expression_types());
        BOOST_REQUIRE_EQUAL(expression_types.size(), 1u);
        BOOST_CHECK_EQUAL(expression_types[0], fern::ExpressionType(
            fern::DataTypes::CONSTANT, fern::ValueTypes::FLOAT64));
    }
}


BOOST_FIXTURE_TEST_CASE(visit_name, Support)
{
    std::shared_ptr<fl::ModuleVertex> tree =
        _xml_parser.parse_string(_algebra_parser.parse_string("a"));
    tree->Accept(_visitor);

    BOOST_CHECK_EQUAL(tree->source_name(), "<string>");
    BOOST_CHECK_EQUAL(tree->line(), 0);
    BOOST_CHECK_EQUAL(tree->col(), 0);
    BOOST_CHECK_EQUAL(tree->scope()->statements().size(), 1u);

    std::shared_ptr<fl::StatementVertex> const& statement(
        tree->scope()->statements()[0]);
    BOOST_REQUIRE(statement);

    fl::NameVertex const* name_vertex(
        dynamic_cast<fl::NameVertex*>(statement.get()));
    BOOST_REQUIRE(name_vertex);

    fl::ExpressionTypes expression_types(name_vertex->expression_types());
    BOOST_REQUIRE_EQUAL(expression_types.size(), 1u);
    BOOST_CHECK_EQUAL(expression_types[0], fern::ExpressionType());
}


BOOST_FIXTURE_TEST_CASE(visit_operation, Support)
{
    {
        std::shared_ptr<fl::ModuleVertex> tree =
            _xml_parser.parse_string(_algebra_parser.parse_string("abs(a)"));
        tree->Accept(_visitor);

        BOOST_CHECK_EQUAL(tree->source_name(), "<string>");
        BOOST_CHECK_EQUAL(tree->line(), 0);
        BOOST_CHECK_EQUAL(tree->col(), 0);

        BOOST_REQUIRE_EQUAL(tree->scope()->statements().size(), 1u);
        std::shared_ptr<fl::StatementVertex> const& statement(
            tree->scope()->statements()[0]);
        BOOST_REQUIRE(statement);
        fl::OperationVertex const* function_call_vertex(
            dynamic_cast<fl::OperationVertex*>(statement.get()));
        BOOST_REQUIRE(function_call_vertex);

        fl::OperationPtr const& operation(
            function_call_vertex->operation());
        BOOST_REQUIRE(operation);

        BOOST_CHECK_EQUAL(operation->parameters().size(), 1u);
        std::vector<fl::Parameter> const& parameters(
            operation->parameters());
        fl::Parameter const& parameter(parameters[0]);
        BOOST_CHECK_EQUAL(parameter.expression_types().size(), 1u);
        BOOST_CHECK_EQUAL(parameter.expression_types()[0].data_type(),
            fern::DataTypes::CONSTANT | fern::DataTypes::STATIC_FIELD);
        BOOST_CHECK_EQUAL(parameter.expression_types()[0].value_type(),
            fern::ValueTypes::NUMBER);

        BOOST_CHECK_EQUAL(operation->results().size(), 1u);
        std::vector<fl::Result> const& results(operation->results());
        fl::Result const& result(results[0]);
        BOOST_CHECK_EQUAL(result.expression_type().data_type(),
            fern::DataTypes::ALL);
        BOOST_CHECK_EQUAL(result.expression_type().value_type(),
            fern::ValueTypes::NUMBER);

        fl::ExpressionTypes expression_types(
            function_call_vertex->expression_types());
        BOOST_REQUIRE_EQUAL(expression_types.size(), 1u);
        BOOST_CHECK_EQUAL(expression_types[0], fern::ExpressionType(
            fern::DataTypes::ALL, fern::ValueTypes::NUMBER));


        // abs(5), abs(-5), abs(5.5)
    }

    // Test whether result data type of read is propagated to result type of
    // abs. Read results in field, so abs must result in field, with the same
    // value types.
    {
        std::shared_ptr<fl::ModuleVertex> tree =
            _xml_parser.parse_string(_algebra_parser.parse_string(
                "abs(read(\"a\"))"));
        tree->Accept(_visitor);

        BOOST_CHECK_EQUAL(tree->source_name(), "<string>");
        BOOST_CHECK_EQUAL(tree->line(), 0);
        BOOST_CHECK_EQUAL(tree->col(), 0);

        BOOST_REQUIRE_EQUAL(tree->scope()->statements().size(), 1u);
        std::shared_ptr<fl::StatementVertex> const& statement(
            tree->scope()->statements()[0]);
        BOOST_REQUIRE(statement);
        fl::OperationVertex const* abs_vertex(
            dynamic_cast<fl::OperationVertex*>(statement.get()));
        BOOST_REQUIRE(abs_vertex);

        fl::ExpressionTypes expression_types(
            abs_vertex->expression_types());
        BOOST_REQUIRE_EQUAL(expression_types.size(), 1u);
        BOOST_CHECK_EQUAL(expression_types[0], fern::ExpressionType(
            fern::DataTypes::STATIC_FIELD, fern::ValueTypes::NUMBER));
    }
}


class OperationResultTypeTester
{

public:

    OperationResultTypeTester(
        fl::AlgebraParser const& algebra_parser,
        fl::XmlParser const& xml_parser,
        fl::AnnotateVisitor& visitor)

        : _algebra_parser(algebra_parser),
          _xml_parser(xml_parser),
          _visitor(visitor)

    {
    }

    void operator()(
        std::string const& script,
        fern::ExpressionType const& expression_type)
    {
        std::shared_ptr<fl::ModuleVertex> tree(_xml_parser.parse_string(
            _algebra_parser.parse_string(script)));
        tree->Accept(_visitor);
        fl::ExpressionVertex* expression_vertex =
            dynamic_cast<fl::ExpressionVertex*>(
                tree->scope()->statements()[0].get());

        fl::ExpressionTypes expression_types(
            expression_vertex->expression_types());
        BOOST_REQUIRE_EQUAL(expression_types.size(), 1u);
        BOOST_CHECK_EQUAL(expression_types[0], expression_type);
    }

private:

    fl::AlgebraParser const& _algebra_parser;

    fl::XmlParser const& _xml_parser;

    fl::AnnotateVisitor& _visitor;

};


BOOST_FIXTURE_TEST_CASE(visit_operation_2, Support)
{
    OperationResultTypeTester tester(_algebra_parser, _xml_parser, _visitor);

    // TODO Update tester from operation to expression.

    // Default integer type is int64.
    tester("5", fern::ExpressionType(
        fern::DataTypes::CONSTANT,
        fern::ValueTypes::INT64));

    // Default float type is float64.
    tester("5.5", fern::ExpressionType(
        fern::DataTypes::CONSTANT,
        fern::ValueTypes::FLOAT64));

    tester("abs(a)", fern::ExpressionType(
        fern::DataTypes::ALL,
        fern::ValueTypes::NUMBER));
    tester("abs(5)", fern::ExpressionType(
        fern::DataTypes::CONSTANT,
        fern::ValueTypes::INT64));
    tester("abs(5.5)", fern::ExpressionType(
        fern::DataTypes::CONSTANT,
        fern::ValueTypes::FLOAT64));

    tester("int32(5)", fern::ExpressionType(
        fern::DataTypes::CONSTANT,
        fern::ValueTypes::INT32));
    tester("int32(5.5)", fern::ExpressionType(
        fern::DataTypes::CONSTANT,
        fern::ValueTypes::INT32));

    tester("5 + 6", fern::ExpressionType(
        fern::DataTypes::CONSTANT,
        fern::ValueTypes::INT64));
    tester("5 + int32(6)", fern::ExpressionType(
        fern::DataTypes::CONSTANT,
        fern::ValueTypes::INT64));
    tester("int32(5) + int32(6)", fern::ExpressionType(
        fern::DataTypes::CONSTANT,
        fern::ValueTypes::INT32));
}


BOOST_FIXTURE_TEST_CASE(visit_attribute, Support)
{
    {
        // TODO hier verder
        // std::shared_ptr<fl::ModuleVertex> tree =
        //     _xml_parser.parse_string(_algebra_parser.parse_string("5"));
        // tree->Accept(_visitor);

        // BOOST_CHECK_EQUAL(tree->source_name(), "<string>");
        // BOOST_CHECK_EQUAL(tree->line(), 0);
        // BOOST_CHECK_EQUAL(tree->col(), 0);
        // BOOST_CHECK_EQUAL(tree->scope()->statements().size(), 1u);

        // std::shared_ptr<fl::StatementVertex> const& statement(
        //     tree->scope()->statements()[0]);
        // BOOST_REQUIRE(statement);

        // fl::NumberVertex<int64_t> const* number_vertex(
        //     dynamic_cast<fl::NumberVertex<int64_t>*>(statement.get()));
        // BOOST_REQUIRE(number_vertex);

        // fl::ExpressionTypes expression_types(
        //     number_vertex->expression_types());
        // BOOST_REQUIRE_EQUAL(expression_types.size(), 1u);
        // BOOST_CHECK_EQUAL(expression_types[0], fern::ExpressionType(
        //     fern::DataTypes::CONSTANT, fern::ValueTypes::INT64));
    }
}

BOOST_AUTO_TEST_SUITE_END()
