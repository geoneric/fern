#include "ranally/operation/operation.h"
#include "ranally/operation/parameter.h"
#include "ranally/operation/result.h"


namespace ranally {

Operation::Operation(
    String const& name,
    String const& description,
    std::vector<Parameter> const& parameters,
    std::vector<Result> const& results)

    : _name(name),
      _description(description),
      _parameters(parameters),
      _results(results)

{
    assert(!name.is_empty());
    assert(!description.is_empty());
}


Operation::Operation(
    Operation const& other)

    : _name(other._name),
      _description(other._description),
      _parameters(other._parameters),
      _results(other._results)

{
}


Operation& Operation::operator=(
    Operation const& other)
{
    if(&other != this) {
        _name = other._name;
        _description = other._description;
        _parameters = other._parameters;
        _results = other._results;
    }

    return *this;
}


String const& Operation::name() const
{
    return _name;
}


String const& Operation::description() const
{
    return _description;
}


size_t Operation::arity() const
{
    return _parameters.size();
}


std::vector<Parameter> const& Operation::parameters() const
{
    return _parameters;
}


std::vector<Result> const& Operation::results() const
{
    return _results;
}

} // namespace ranally
