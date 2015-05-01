// -----------------------------------------------------------------------------
// Fern © Geoneric
//
// This file is part of Geoneric Fern which is available under the terms of
// the GNU General Public License (GPL), version 2. If you do not want to
// be bound by the terms of the GPL, you may purchase a proprietary license
// from Geoneric (http://www.geoneric.eu/contact).
// -----------------------------------------------------------------------------
#pragma once
#include "fern/core/path.h"


namespace fern {
namespace language {

bool               file_exists         (Path const& path);

bool               file_is_writable    (Path const& path);

bool               directory_is_writable(
                                        Path const& path);

void               write_file          (std::string const& value,
                                        Path const& path);

} // namespace language
} // namespace fern
