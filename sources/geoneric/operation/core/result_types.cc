#include "geoneric/operation/core/result_types.h"


namespace geoneric {

ResultTypes::ResultTypes(
    size_t size)

    : std::vector<ResultType>(size)

{
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


} // namespace geoneric
