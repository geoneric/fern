#include "fern/core/detail/function_wrapper.h"


namespace fern {
namespace detail {

FunctionWrapper::FunctionWrapper(
    FunctionWrapper&& other)

    : _concept(std::move(other._concept))

{
}


FunctionWrapper& FunctionWrapper::operator=(
    FunctionWrapper&& other)
{
    if(&other != this) {
        _concept = std::move(other._concept);
    }

    return *this;
}


void FunctionWrapper::operator()()
{
    _concept->call();
}

} // namespace detail
} // namespace fern
