// -----------------------------------------------------------------------------
// Fern © Geoneric
//
// This file is part of Geoneric Fern which is available under the terms of
// the GNU General Public License (GPL), version 2. If you do not want to
// be bound by the terms of the GPL, you may purchase a proprietary license
// from Geoneric (http://www.geoneric.eu/contact).
// -----------------------------------------------------------------------------
#pragma once
#include "fern/python/feature/masked_raster.h"


namespace fern {
namespace python {

MaskedRasterHandle read_raster         (std::string const& name);

void               write_raster        (MaskedRasterHandle const& raster,
                                        std::string const& name,
                                        std::string const& format);

} // namespace python
} // namespace fern
