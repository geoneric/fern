// -----------------------------------------------------------------------------
// Fern © Geoneric
//
// This file is part of Geoneric Fern which is available under the terms of
// the GNU General Public License (GPL), version 2. If you do not want to
// be bound by the terms of the GPL, you may purchase a proprietary license
// from Geoneric (http://www.geoneric.eu/contact).
// -----------------------------------------------------------------------------
#include "fern/language/operation/core/operation.h"
#include <cassert>
#include "fern/language/operation/core/parameter.h"
#include "fern/language/operation/core/result.h"


namespace fern {
namespace language {

Operation::Operation(
    std::string const& name,
    std::string const& description,
    std::vector<Parameter> const& parameters,
    std::vector<Result> const& results)

    : _name(name),
      _description(description),
      _parameters(parameters),
      _results(results)

{
    assert(!name.empty());
    assert(!description.empty());
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


// std::string Operation::xml() const
// {
//     std::string result = boost::format(
//         "<?xml version="1.0"?>\n"
//         "  <Operation>\n"
//         "  <Name>%1%</Name>\n"
//         "  <Description>%2%</Description>\n"
//         "  <Parameters>\n"
//         "    %3%\n"
//         "  </Parameters>\n"
//         "  <Results>\n"
//         "    %4%\n"
//         "  </Results>\n"
//         "</Operation>\n"
//         ) % name() % description % parameters % results;
// 
//     return result;
// }


std::string const& Operation::name() const
{
    return _name;
}


std::string const& Operation::description() const
{
    return _description;
}


size_t Operation::arity() const
{
    return _parameters.size();
}


ExpressionType Operation::expression_type(
#ifndef NDEBUG
    size_t index,
#else
    size_t /* index */,
#endif
    std::vector<ExpressionType> const& /* argument_types */) const
{
    assert(_results.size() == 0);
    assert(index == 0);

    return ExpressionType();
}


std::vector<Parameter> const& Operation::parameters() const
{
    return _parameters;
}


std::vector<Result> const& Operation::results() const
{
    return _results;
}

} // namespace language
} // namespace fern
