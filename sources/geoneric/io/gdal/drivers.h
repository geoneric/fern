#pragma once
#include <memory>
#include <vector>
#include "geoneric/io/gdal/dataset.h"
#include "geoneric/io/gdal/driver.h"


namespace geoneric {

extern std::vector<std::shared_ptr<Driver>> drivers;

std::shared_ptr<Dataset> open          (String const& name);

} // namespace geoneric
