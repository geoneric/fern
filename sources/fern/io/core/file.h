#pragma once
#include "fern/core/path.h"


namespace fern {

bool               file_exists         (Path const& path);

bool               file_is_writable    (Path const& path);

bool               directory_is_writable(
                                        Path const& path);

void               write_file          (String const& value,
                                        Path const& path);

} // namespace fern
