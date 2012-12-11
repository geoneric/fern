#include "ranally/language/result_types.h"


namespace ranally {

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
