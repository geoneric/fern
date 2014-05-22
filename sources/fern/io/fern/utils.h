#pragma once
#include <memory>
#include <H5Cpp.h>
#include "fern/core/path.h"
#include "fern/io/core/open_mode.h"


namespace fern {

std::shared_ptr<H5::H5File> open_file  (Path const& path,
                                        OpenMode open_mode);

} // namespace fern
