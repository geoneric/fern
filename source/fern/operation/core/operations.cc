// -----------------------------------------------------------------------------
// Fern © Geoneric
//
// This file is part of Geoneric Fern which is available under the terms of
// the GNU General Public License (GPL), version 2. If you do not want to
// be bound by the terms of the GPL, you may purchase a proprietary license
// from Geoneric (http://www.geoneric.eu/contact).
// -----------------------------------------------------------------------------
#include "fern/operation/core/operations.h"


namespace fern {

Operations::Operations(
    std::initializer_list<OperationPtr> values)

    : _operations()

{
    for(auto value: values) {
        _operations[value->name()] = value;
    }
}


bool Operations::empty() const
{
    return _operations.empty();
}


size_t Operations::size() const
{
    return _operations.size();
}


bool Operations::has_operation(
    String const& name) const
{
    return _operations.find(name) != _operations.end();
}


OperationPtr const& Operations::operation(
    String const& name) const
{
    std::map<String, OperationPtr>::const_iterator it =
        _operations.find(name);
    assert(it != _operations.end());
    return (*it).second;
}

} // namespace fern
