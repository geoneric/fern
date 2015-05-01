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
#include <map>
// #include "fern/core/data_name.h"
#include "fern/language/io/core/dataset.h"
#include "fern/language/io/core/driver.h"


namespace fern {
namespace language {

extern std::map<std::string, std::shared_ptr<Driver>> drivers;

bool               dataset_exists      (std::string const& name,
                                        OpenMode open_mode,
                                        std::string const& format="");

std::shared_ptr<Dataset> open_dataset  (std::string const& name,
                                        OpenMode open_mode,
                                        std::string const& format="");

// ExpressionType     expression_type     (DataName const& data_name);

// std::shared_ptr<Driver> driver_for     (Path const& path,
//                                         OpenMode open_mode);

// std::shared_ptr<Dataset> create_dataset(Attribute const& attribute,
//                                         std::string const& name,
//                                         std::string const& format);

} // namespace language
} // namespace fern
