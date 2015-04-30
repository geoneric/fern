// -----------------------------------------------------------------------------
// Fern © Geoneric
//
// This file is part of Geoneric Fern which is available under the terms of
// the GNU General Public License (GPL), version 2. If you do not want to
// be bound by the terms of the GPL, you may purchase a proprietary license
// from Geoneric (http://www.geoneric.eu/contact).
// -----------------------------------------------------------------------------
#include "fern/language/operation/core/expression_types.h"


namespace fern {

ExpressionTypes::ExpressionTypes(
    size_t size)

    : std::vector<ExpressionType>(size)

{
}


ExpressionTypes::ExpressionTypes(
    std::initializer_list<ExpressionType> const& expression_types)

    : std::vector<ExpressionType>(expression_types)

{
}


bool ExpressionTypes::is_satisfied_by(
    ExpressionType const& expression_type) const
{
    bool result = false;

    for(auto const& this_result_type: *this) {
        if(this_result_type.is_satisfied_by(expression_type)) {
            result = true;
            break;
        }
    }

    return result;
}


size_t ExpressionTypes::id_of_satisfying_type(
    ExpressionTypes const& expression_types) const
{
    size_t result = expression_types.size();

    for(size_t i = 0; i < expression_types.size(); ++i) {
        if(is_satisfied_by(expression_types[i])) {
            result = i;
            break;
        }
    }

    return result;
}


//! Return whether \a expression_types satisfies this instance.
/*!
  \tparam    .
  \param     .
  \return    .
  \exception .
  \warning   .
  \sa        .

  \a expression_types satisfies this instance if one of the instances in the
  \a expression_types collection is a subset of this instance.
*/
bool ExpressionTypes::is_satisfied_by(
    ExpressionTypes const& expression_types) const
{
    return id_of_satisfying_type(expression_types) < expression_types.size();
}


bool ExpressionTypes::fixed() const
{
    bool result = empty() ? false : true;

    for(ExpressionType const& expression_type: *this) {
        if(!expression_type.fixed()) {
            result = false;
            break;
        }
    }

    return result;
}


bool ExpressionTypes::is_empty() const
{
    return empty();
}


size_t ExpressionTypes::size() const
{
    return std::vector<ExpressionType>::size();
}


ExpressionType const& ExpressionTypes::operator[](
    size_t index) const
{
    assert(index < size());
    return std::vector<ExpressionType>::operator[](index);
}


ExpressionType& ExpressionTypes::operator[](
    size_t index)
{
    assert(index < size());
    return std::vector<ExpressionType>::operator[](index);
}


ExpressionTypes::const_iterator ExpressionTypes::begin() const
{
    return std::vector<ExpressionType>::begin();
}


ExpressionTypes::const_iterator ExpressionTypes::end() const
{
    return std::vector<ExpressionType>::end();
}


void ExpressionTypes::add(
    ExpressionType const& expression_type)
{
    emplace_back(expression_type);
}


std::ostream& operator<<(
    std::ostream& stream,
    ExpressionTypes const& expression_types)
{
    if(!expression_types.is_empty()) {
        stream << expression_types[0];

        for(size_t i = 1; i < expression_types.size(); ++i) {
            stream << " | " << expression_types[i];
        }
    }

    return stream;
}

} // namespace fern
