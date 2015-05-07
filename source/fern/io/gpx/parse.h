// -----------------------------------------------------------------------------
// Fern © Geoneric
//
// This file is part of Geoneric Fern which is available under the terms of
// the GNU General Public License (GPL), version 2. If you do not want to
// be bound by the terms of the GPL, you may purchase a proprietary license
// from Geoneric (http://www.geoneric.eu/contact).
// -----------------------------------------------------------------------------
#pragma once
#include "fern/io/gpx/gpx-1.0.hxx"
#include "fern/io/gpx/gpx-1.1.hxx"


namespace fern {
namespace io {
namespace gpx_1_0 {

std::unique_ptr<::gpx_1_0::gpx>
                   parse               (std::string const& pathname);

} // namespace gpx_1_0


namespace gpx_1_1 {

std::unique_ptr<::gpx_1_1::gpxType>
                   parse               (std::string const& pathname);

} // namespace gpx_1_1
} // namespace io
} // namespace fern
