#include "ranally/operation/result_types.h"


namespace ranally {

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


} // namespace ranally
