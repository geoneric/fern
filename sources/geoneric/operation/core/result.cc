#include "geoneric/operation/core/result.h"
#include <cassert>


namespace geoneric {

Result::Result(
    String const& name,
    String const& description,
    ResultType const& result_type)

    : _name(name),
      _description(description),
      _result_type(result_type)

{
    assert(!_name.is_empty());
    assert(!_description.is_empty());
}


Result::Result(
    Result const& other)

    : _name(other._name),
      _description(other._description),
      _result_type(other._result_type)

{
}


Result& Result::operator=(
  Result const& other)
{
    if(&other != this) {
        _name = other._name;
        _description = other._description;
        _result_type = other._result_type;
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


ResultType const& Result::result_type() const
{
    return _result_type;
}

} // namespace geoneric
