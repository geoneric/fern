#include "Ranally/Operation/Operation.h"
#include "Ranally/Operation/Parameter.h"
#include "Ranally/Operation/Result.h"


namespace ranally {
namespace operation {

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
    assert(!name.isEmpty());
    assert(!description.isEmpty());
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


Operation::~Operation()
{
}


String const& Operation::name() const
{
    return _name;
}


String const& Operation::description() const
{
    return _description;
}


std::vector<Parameter> const& Operation::parameters() const
{
    return _parameters;
}


std::vector<Result> const& Operation::results() const
{
    return _results;
}

} // namespace operation
} // namespace ranally
