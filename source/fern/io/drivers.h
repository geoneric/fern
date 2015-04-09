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
#include "fern/io/core/dataset.h"
#include "fern/io/core/driver.h"


namespace fern {

extern std::map<String, std::shared_ptr<Driver>> drivers;

bool               dataset_exists      (String const& name,
                                        OpenMode open_mode,
                                        String const& format="");

std::shared_ptr<Dataset> open_dataset  (String const& name,
                                        OpenMode open_mode,
                                        String const& format="");

// ExpressionType     expression_type     (DataName const& data_name);

// std::shared_ptr<Driver> driver_for     (Path const& path,
//                                         OpenMode open_mode);

// std::shared_ptr<Dataset> create_dataset(Attribute const& attribute,
//                                         String const& name,
//                                         String const& format);

} // namespace fern
