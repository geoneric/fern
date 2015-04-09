// -----------------------------------------------------------------------------
// Fern © Geoneric
//
// This file is part of Geoneric Fern which is available under the terms of
// the GNU General Public License (GPL), version 2. If you do not want to
// be bound by the terms of the GPL, you may purchase a proprietary license
// from Geoneric (http://www.geoneric.eu/contact).
// -----------------------------------------------------------------------------
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
