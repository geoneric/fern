#pragma once
#include <memory>
#include <cpp/H5Cpp.h>
#include "geoneric/core/path.h"
#include "geoneric/io/core/open_mode.h"


namespace geoneric {

std::shared_ptr<H5::H5File> open_file  (Path const& path,
                                        OpenMode open_mode);

} // namespace geoneric
