#include "geoneric/operation/core/result_types.h"


namespace geoneric {

ResultTypes::ResultTypes(
    size_t size)

    : std::vector<ResultType>(size)

{
}


ResultTypes::ResultTypes(
    std::initializer_list<ResultType> const& result_types)

    : std::vector<ResultType>(result_types)

{
}


bool ResultTypes::is_satisfied_by(
    ResultType const& result_type) const
{
    bool result = false;

    for(auto const& this_result_type: *this) {
        if(this_result_type.is_satisfied_by(result_type)) {
            result = true;
            break;
        }
    }

    return result;
}


size_t ResultTypes::id_of_satisfying_type(
    ResultTypes const& result_types) const
{
    size_t result = result_types.size();

    for(size_t i = 0; i < result_types.size(); ++i) {
        if(is_satisfied_by(result_types[i])) {
            result = i;
            break;
        }
    }

    return result;
}


//! Return whether \a result_types satisfies this instance.
/*!
  \tparam    .
  \param     .
  \return    .
  \exception .
  \warning   .
  \sa        .

  \a result_types satisfies this instance if one of the instances in the
  \a result_types collection is a subset of this instance.
*/
bool ResultTypes::is_satisfied_by(
    ResultTypes const& result_types) const
{
    return id_of_satisfying_type(result_types) < result_types.size();
}


bool ResultTypes::fixed() const
{
    bool result = empty() ? false : true;

    for(ResultType const& result_type: *this) {
        if(!result_type.fixed()) {
            result = false;
            break;
        }
    }

    return result;
}


std::ostream& operator<<(
    std::ostream& stream,
    ResultTypes const& result_types)
{
    if(!result_types.empty()) {
        stream << result_types[0];

        for(size_t i = 1; i < result_types.size(); ++i) {
            stream << " | " << result_types[i];
        }
    }

    return stream;
}

} // namespace geoneric
