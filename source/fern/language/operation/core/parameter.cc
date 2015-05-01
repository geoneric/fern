// -----------------------------------------------------------------------------
// Fern © Geoneric
//
// This file is part of Geoneric Fern which is available under the terms of
// the GNU General Public License (GPL), version 2. If you do not want to
// be bound by the terms of the GPL, you may purchase a proprietary license
// from Geoneric (http://www.geoneric.eu/contact).
// -----------------------------------------------------------------------------
#include "fern/language/operation/core/parameter.h"
#include <cassert>


namespace fern {
namespace language {

//!
/*!
  \tparam    .
  \param     .
  \return    .
  \exception .
  \warning   .
  \sa        .

  The \a data_types argument passed in can not be DataTypes::UNKNOWN. The
  \a value_types argument passed in can be ValueTypes::UNKNOWN, in which case
  the value type is taken to be not relevant. This is the case for operations
  that work on feature domains and not on the attributes, like an operation
  that counts the number of feature elements.
*/
Parameter::Parameter(
    std::string const& name,
    std::string const& description,
    DataTypes data_types,
    ValueTypes value_types)

    : _name(name),
      _description(description),
      _expression_types({ExpressionType(data_types, value_types)})
      // _data_types(data_types),
      // _value_types(value_types)

{
    assert(!_name.empty());
    assert(!_description.empty());
    // assert(_data_types != DataTypes::UNKNOWN);
#ifndef NDEBUG
    for(auto const& expression_type: _expression_types) {
        assert(expression_type.data_type() != DataTypes::UNKNOWN);
    }
#endif
}


Parameter::Parameter(
    Parameter const& other)

    : _name(other._name),
      _description(other._description),
      _expression_types(other._expression_types)
      // _data_types(other._data_types),
      // _value_types(other._value_types)

{
}


Parameter& Parameter::operator=(
    Parameter const& other)
{
    if(&other != this) {
        _name = other._name;
        _description = other._description;
        // _data_types = other._data_types;
        // _value_types = other._value_types;
        _expression_types = other._expression_types;
    }

    return *this;
}


Parameter::~Parameter()
{
}


std::string const& Parameter::name() const
{
    return _name;
}


std::string const& Parameter::description() const
{
    return _description;
}


// DataTypes Parameter::data_types() const
// {
//     return _data_types;
// }
// 
// 
// ValueTypes Parameter::value_types() const
// {
//     return _value_types;
// }


ExpressionTypes Parameter::expression_types() const
{
    return _expression_types;
}

} // namespace language
} // namespace fern
