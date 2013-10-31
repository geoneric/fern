#pragma once
#include "geoneric/core/path.h"


namespace geoneric {

bool               file_exists         (Path const& path);

bool               directory_is_writable(
                                        Path const& path);

} // namespace geoneric
