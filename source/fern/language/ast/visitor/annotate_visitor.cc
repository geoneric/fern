// -----------------------------------------------------------------------------
// Fern © Geoneric
//
// This file is part of Geoneric Fern which is available under the terms of
// the GNU General Public License (GPL), version 2. If you do not want to
// be bound by the terms of the GPL, you may purchase a proprietary license
// from Geoneric (http://www.geoneric.eu/contact).
// -----------------------------------------------------------------------------
#include "fern/language/ast/visitor/annotate_visitor.h"
#include "fern/core/type_traits.h"
#include "fern/language/operation/core/result.h"
#include "fern/language/ast/core/vertices.h"


namespace fern {
namespace language {

AnnotateVisitor::AnnotateVisitor(
    OperationsPtr const& operations)

    : AstVisitor(),
      _operations(operations),
      _stack(),
      _symbol_table()

{
    assert(_operations);
    _symbol_table.push_scope();
}


AnnotateVisitor::~AnnotateVisitor()
{
    _symbol_table.pop_scope();
}


// void AnnotateVisitor::reset()
// {
//     clear_stack();
//     _symbol_table = SymbolTable<ExpressionType>();
//     _symbol_table.push_scope();
// }


std::stack<ExpressionType> const& AnnotateVisitor::stack() const
{
    return _stack;
}


void AnnotateVisitor::clear_stack()
{
    _stack = std::stack<ExpressionType>();
}


//! Add global symbols to the symbol table.
/*!
  \param     symbol_table Symbol table with symbols to add.
  \warning   The symbol table passed in must only contain global symbols
             (scope level 1).

  Use this method to pass knowledge about undefined identifiers from the
  context to the module. This way, the module can be better annotated. If all
  symbols are resolved, annotation must be able to calculate exactly what
  the expression types of the results will be.

  Symbols in the instance's symbol table with the same name as symbols from
  \a symbol_table will be overwritten.
*/
void AnnotateVisitor::add_global_symbols(
    SymbolTable<ExpressionType> const& symbol_table)
{
    assert(_symbol_table.scope_level() > 0u);

    if(!symbol_table.empty()) {
        assert(symbol_table.scope_level() == 1u);

        // Add the symbols to the symbol table.
        for(auto const& pair: symbol_table.scope(1u)) {
            _symbol_table.add_value(pair.first, pair.second);
        }
    }
}


void AnnotateVisitor::Visit(
    AssignmentVertex& vertex)
{
    // Let the source expression execute itself, leaving the result(s) on the
    // stack.
    vertex.expression()->Accept(*this);

    // Assume the target expression is a NameVertex (it should, for now).
    assert(dynamic_cast<NameVertex const*>(vertex.target().get()));

    // Store the result in a scoped symbol table for later reference.
    // Update scope at correct moments in other visit functions.
    assert(!_stack.empty());
    _symbol_table.add_value(vertex.target()->name(), _stack.top());
    _stack.pop();

    // Propagate the result types from the expression to the target.
    ExpressionVertex const& expression(*vertex.expression());
    ExpressionVertex& target(*vertex.target());
    assert(target.expression_types().is_empty());
    assert(expression.expression_types().size() == 1);
    target.set_expression_types(expression.expression_types());
    // vertex.target()->Accept(*this);
}


#define VISIT_NUMBER_VERTEX(                                                   \
    type)                                                                      \
void AnnotateVisitor::Visit(                                                   \
    NumberVertex<type>& vertex)                                                \
{                                                                              \
    assert(vertex.expression_types().is_empty());                              \
    ExpressionType expression_type(DataTypes::CONSTANT,                        \
        TypeTraits<type>::value_types);                                        \
    _stack.push(expression_type);                                              \
    vertex.add_result_type(expression_type);                                   \
    assert(vertex.expression_types().size() == 1);                             \
}

VISIT_NUMBER_VERTICES(VISIT_NUMBER_VERTEX)

#undef VISIT_NUMBER_VERTEX


void AnnotateVisitor::Visit(
    StringVertex& vertex)
{
    assert(vertex.expression_types().is_empty());
    ExpressionType expression_type(DataTypes::CONSTANT, ValueTypes::STRING);
    _stack.push(expression_type);
    vertex.add_result_type(expression_type);
    assert(vertex.expression_types().size() == 1);
}


void AnnotateVisitor::Visit(
    OperationVertex& vertex)
{
    assert(vertex.expression_types().is_empty());

    // Depth first, visit the children first.
    AstVisitor::Visit(vertex);


    // Get the result types from all argument expressions provided.
    std::vector<ExpressionType> argument_types;
    for(size_t i = 0; i < vertex.expressions().size(); ++i) {
        assert(!_stack.empty());
        argument_types.emplace_back(_stack.top());
        _stack.pop();
    }

    ExpressionTypes expression_types(1);

    // Retrieve info about the operation, if available.
    if(_operations->has_operation(vertex.name())) {
        assert(!vertex.operation());
        OperationPtr const& operation(_operations->operation(vertex.name()));
        vertex.set_operation(operation);

        assert(vertex.expression_types().is_empty());

        // http://en.cppreference.com/w/cpp/types/common_type
        // There are vertex.expressions().size() ExpressionType instances on the
        // stack. Given these and the operation, calculate a ExpressionType
        // instance for the result of this expression. It is possible that
        // there are not enough arguments provided. In that case the
        // calculation of the result type may fail. Validation will pick that
        // up.

        if(vertex.expressions().size() == operation->arity()) {
            // Calculate result type for each result.
            // TODO Update for multiple results.
            for(size_t i = 0; i < 1; ++i) {
                ExpressionType expression_type = operation->expression_type(i,
                    argument_types);
                expression_types[i] = expression_type;
            }
        }
    }

    for(auto expression_type: expression_types) {
        _stack.push(expression_type);
    }
    vertex.set_expression_types(expression_types);

    assert(vertex.expression_types().size() == 1);
}


void AnnotateVisitor::Visit(
    NameVertex& vertex)
{
    assert(vertex.expression_types().is_empty());

    ExpressionType expression_type;

    // If the symbol is defined, retrieve the value from the symbol table.
    if(_symbol_table.has_value(vertex.name())) {
        expression_type = _symbol_table.value(vertex.name());
    }

    // Push the value onto the stack, even if it is undefined.
    _stack.push(expression_type);
    vertex.add_result_type(expression_type);

    assert(vertex.expression_types().size() == 1);
}


void AnnotateVisitor::Visit(
    ModuleVertex& vertex)
{
    AstVisitor::Visit(vertex);
}


// void AnnotateVisitor::Visit(
//     SubscriptVertex& /* vertex */)
// {
//     // TODO
//     // switch(_mode) {
//     //     case Using: {
//     //         // The result type of a subscript expression is the same as the
//     //         // result type of the main expression.
//     //         assert(vertex.expression_types().is_empty());
//     //         vertex.set_expression_types(
//     //             vertex.expression()->expression_types());
//     //         break;
//     //     }
//     //     case Defining: {
//     //         break;
//     //     }
//     // }
// }

} // namespace language
} // namespace fern
