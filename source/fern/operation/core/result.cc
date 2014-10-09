#include "fern/operation/core/result.h"
#include <cassert>


namespace fern {

Result::Result(
    String const& name,
    String const& description,
    ExpressionType const& expression_type)

    : _name(name),
      _description(description),
      _expression_type(expression_type)

{
    assert(!_name.is_empty());
    assert(!_description.is_empty());
}


Result::Result(
    Result const& other)

    : _name(other._name),
      _description(other._description),
      _expression_type(other._expression_type)

{
}


Result& Result::operator=(
  Result const& other)
{
    if(&other != this) {
        _name = other._name;
        _description = other._description;
        _expression_type = other._expression_type;
    }

    return *this;
}


String const& Result::name() const
{
    return _name;
}


String const& Result::description() const
{
    return _description;
}


ExpressionType const& Result::expression_type() const
{
    return _expression_type;
}

} // namespace fern
