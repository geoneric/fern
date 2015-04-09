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
#include <tuple>
#include <vector>
#include "fern/interpreter/data_source.h"
#include "fern/interpreter/data_sync.h"
#include "fern/compiler/data_description.h"


namespace fern {

std::tuple<
    std::vector<std::shared_ptr<DataSource>>,
    std::vector<std::shared_ptr<DataSync>>>
                   parse_command_line  (int argc,
                                        char** argv,
                                        std::vector<DataDescription> const&
                                            arguments,
                                        std::vector<DataDescription> const&
                                            results);

} // Namespace fern
