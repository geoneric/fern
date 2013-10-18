#pragma once
#include <memory>
#include <map>
#include "geoneric/io/core/dataset.h"
#include "geoneric/io/core/driver.h"


namespace geoneric {

extern std::map<String, std::shared_ptr<Driver>> drivers;

bool               dataset_exists      (String const& name,
                                        OpenMode open_mode,
                                        String const& format="");

std::shared_ptr<Dataset> open_dataset  (String const& name,
                                        OpenMode open_mode,
                                        String const& format="");

std::shared_ptr<Dataset> create_dataset(Attribute const& attribute,
                                        String const& name,
                                        String const& format);

} // namespace geoneric
