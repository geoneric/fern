// -----------------------------------------------------------------------------
// Fern © Geoneric
//
// This file is part of Geoneric Fern which is available under the terms of
// the GNU General Public License (GPL), version 2. If you do not want to
// be bound by the terms of the GPL, you may purchase a proprietary license
// from Geoneric (http://www.geoneric.eu/contact).
// -----------------------------------------------------------------------------
#include "fern/language/ast/core/expression_vertex.h"


namespace fern {
namespace language {

ExpressionVertex::ExpressionVertex(
    std::string const& name)

    : StatementVertex(),
      _name(name)

{
}


ExpressionVertex::ExpressionVertex(
    int line_nr,
    int col_id,
    std::string const& name)

    : StatementVertex(line_nr, col_id),
      _name(name)

{
}


std::string const& ExpressionVertex::name() const
{
    return _name;
}


void ExpressionVertex::set_expression_types(
    ExpressionTypes const& expression_types)
{
    _expression_types = expression_types;
}


void ExpressionVertex::add_result_type(
    ExpressionType const& expression_type)
{
    _expression_types.add(expression_type);
}


ExpressionTypes const& ExpressionVertex::expression_types() const
{
    return _expression_types;
}


void ExpressionVertex::set_value(
    ExpressionVertexPtr const& value)
{
    _value = value;
}


ExpressionVertexPtr const& ExpressionVertex::value() const
{
    return _value;
}

} // namespace language
} // namespace fern
