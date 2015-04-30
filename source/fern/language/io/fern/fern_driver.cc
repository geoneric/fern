// -----------------------------------------------------------------------------
// Fern © Geoneric
//
// This file is part of Geoneric Fern which is available under the terms of
// the GNU General Public License (GPL), version 2. If you do not want to
// be bound by the terms of the GPL, you may purchase a proprietary license
// from Geoneric (http://www.geoneric.eu/contact).
// -----------------------------------------------------------------------------
#include "fern/language/io/fern/fern_driver.h"
#include "fern/core/data_name.h"
#include "fern/language/feature/visitor/attribute_type_visitor.h"
#include "fern/language/io/core/file.h"
#include "fern/language/io/fern/fern_dataset.h"
#include "fern/language/io/fern/utils.h"


namespace fern {

FernDriver::FernDriver()

    : Driver("Fern")

{
}


bool FernDriver::can_open_for_read(
    std::string const& name)
{
    bool result = false;

    try {
        open_file(name, OpenMode::READ);
        result = true;
    }
    catch(...) {
    }

    return result;
}


bool FernDriver::can_open_for_update(
    std::string const& name)
{
    bool result = false;

    try {
        open_file(name, OpenMode::UPDATE);
        result = true;
    }
    catch(...) {
    }

    return result;
}


bool FernDriver::can_open_for_overwrite(
    std::string const& name)
{
    bool result = false;

    try {
        open_file(name, OpenMode::OVERWRITE);
        result = true;
    }
    catch(...) {
    }

    return result;

    // return (file_exists(name) && can_open_for_update(name)) ||
    //     directory_is_writable(Path(name).parent_path());
}


bool FernDriver::can_open(
    std::string const& name,
    OpenMode open_mode)
{
    bool result = false;

    switch(open_mode) {
        case OpenMode::READ: {
            result = can_open_for_read(name);
            break;
        }
        case OpenMode::OVERWRITE: {
            result = can_open_for_overwrite(name);
            break;
        }
        case OpenMode::UPDATE: {
            result = can_open_for_update(name);
            break;
        }
    }

    return result;
}


std::shared_ptr<Dataset> FernDriver::open(
    std::string const& name,
    OpenMode open_mode)
{
    return std::shared_ptr<Dataset>(std::make_shared<FernDataset>(name,
        open_mode));
}

} // namespace fern
