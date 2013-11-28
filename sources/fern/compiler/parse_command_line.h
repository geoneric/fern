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
