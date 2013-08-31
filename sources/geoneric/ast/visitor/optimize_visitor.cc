#include "geoneric/ast/visitor/optimize_visitor.h"
#include <algorithm>
#include "geoneric/core/string.h"
#include "geoneric/ast/core/assignment_vertex.h"


namespace geoneric {

OptimizeVisitor::OptimizeVisitor()

    : Visitor(),
      _mode(Mode::Defining)

{
}


void OptimizeVisitor::register_expression_for_inlining(
    ExpressionVertex const* use,
    ExpressionVertexPtr const& expression)
{
    _inline_expressions[use] = expression;
}


void OptimizeVisitor::visit_statements(
    StatementVertices& statements)
{
    Visitor::visit_statements(statements);

    switch(_mode) {
        case Mode::Using: {
            break;
        }
        case Mode::Defining: {
            std::vector<size_t> statements_to_erase,
                superfluous_statements_to_erase;

            for(size_t i = 0; i < statements.size(); ++i) {
                for(size_t j = 0; j < _superfluous_statements.size(); ++j) {
                    if(_superfluous_statements[j] == statements[i].get()) {
                        statements_to_erase.push_back(i);
                        superfluous_statements_to_erase.push_back(j);
                    }
                }
            }

            std::reverse(statements_to_erase.begin(),
                statements_to_erase.end());
            for(size_t i = 0; i < statements_to_erase.size(); ++i) {
                statements.erase(statements.begin() + statements_to_erase[i]);
            }

            std::sort(superfluous_statements_to_erase.begin(),
                superfluous_statements_to_erase.end());
            std::reverse(superfluous_statements_to_erase.begin(),
                superfluous_statements_to_erase.end());
            for(size_t i = 0; i < superfluous_statements_to_erase.size(); ++i) {
                _superfluous_statements.erase(_superfluous_statements.begin() +
                    superfluous_statements_to_erase[i]);
            }

            break;
        }
    }
}


void OptimizeVisitor::Visit(
    AssignmentVertex& vertex)
{
    // Inline the defining expression, if possible.
    switch(_mode) {
        case Mode::Using: {
            vertex.expression()->Accept(*this);

            std::map<ExpressionVertex const*, ExpressionVertexPtr>::iterator
                it = _inline_expressions.find(vertex.expression().get());
            if(it != _inline_expressions.end()) {
std::cout << "inserting " << (*it).second->name().encode_in_utf8() << std::endl;
                // Schedule the defining statement for removal.
                _inlined_expressions.push_back((*it).second);
                vertex.set_expression((*it).second);
                _inline_expressions.erase(it);
            }

            break;
        }
        case Mode::Defining: {
            vertex.target()->Accept(*this);

            std::vector<ExpressionVertexPtr>::iterator it = std::find(
                _inlined_expressions.begin(), _inlined_expressions.end(),
                    vertex.expression());
            if(it != _inlined_expressions.end()) {
                _superfluous_statements.push_back(&vertex);
                _inlined_expressions.erase(it);
            }

            break;
        }
    }
}


void OptimizeVisitor::Visit(
    NameVertex& vertex)
{
    switch(_mode) {
        case Mode::Using: {
            std::vector<NameVertex*> const& definitions(vertex.definitions());

            if(definitions.size() == 1 && definitions[0]->uses().size() == 1) {
                // This identifier has one definition and the defining
                // identifier is used only here.
                assert(definitions[0]->uses()[0] == &vertex);

                // Register the value of the defining expression for
                // inlining at the use location.
std::cout << "register inlining of " << vertex.name().encode_in_utf8() << " by " << definitions[0]->value()->name().encode_in_utf8() << std::endl;
                register_expression_for_inlining(&vertex,
                    definitions[0]->value());
            }

            break;
        }
        case Mode::Defining: {
            break;
        }
    }
}


void OptimizeVisitor::Visit(
    ModuleVertex& vertex)
{
    bool inlined_expressions;

    do {
std::cout << "visit script" << std::endl;
        assert(_inline_expressions.empty());
        assert(_inlined_expressions.empty());
        assert(_superfluous_statements.empty());

        // First visit all use locations of name vertices.
        _mode = Mode::Using;
        Visitor::Visit(vertex);

        // If expressions have been inlined, then the script will
        // change. We need to repeat the visit until the script is stable.
        inlined_expressions = !_inlined_expressions.empty();

        // Now visit all defining locations of name vertices.
        _mode = Mode::Defining;
        Visitor::Visit(vertex);

        assert(_inline_expressions.empty());
        assert(_inlined_expressions.empty());
        assert(_superfluous_statements.empty());
    } while(inlined_expressions);
}

} // namespace geoneric
