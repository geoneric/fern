// -----------------------------------------------------------------------------
// Fern © Geoneric
//
// This file is part of Geoneric Fern which is available under the terms of
// the GNU General Public License (GPL), version 2. If you do not want to
// be bound by the terms of the GPL, you may purchase a proprietary license
// from Geoneric (http://www.geoneric.eu/contact).
// -----------------------------------------------------------------------------
#pragma once
#include <memory>
#include "fern/core/path.h"
#include "fern/language/io/core/open_mode.h"
#include "fern/language/io/fern/hdf5_file.h"


namespace fern {
namespace language {

std::unique_ptr<HDF5File>
                   open_file           (Path const& path,
                                        OpenMode open_mode);

} // namespace language
} // namespace fern
