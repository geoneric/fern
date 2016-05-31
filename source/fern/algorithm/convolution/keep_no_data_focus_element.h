// -----------------------------------------------------------------------------
// Fern © Geoneric
//
// This file is part of Geoneric Fern which is available under the terms of
// the GNU General Public License (GPL), version 2. If you do not want to
// be bound by the terms of the GPL, you may purchase a proprietary license
// from Geoneric (http://www.geoneric.eu/contact).
// -----------------------------------------------------------------------------
#pragma once


namespace fern {
namespace algorithm {
namespace convolve {

/*!
    @brief      NoDataFocusElementPolicy which results in no-data focus
                elements beіng copied to the result.
    @sa         @ref fern_algorithm_convolution_policies
*/
class KeepNoDataFocusElement
{

public:

    static bool const keep_no_data = true;

};

} // namespace convolve
} // namespace algorithm
} // namespace fern
