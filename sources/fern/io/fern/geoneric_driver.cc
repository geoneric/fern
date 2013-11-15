#include "fern/io/fern/geoneric_driver.h"
#include "fern/core/data_name.h"
#include "fern/feature/visitor/attribute_type_visitor.h"
#include "fern/io/core/file.h"
#include "fern/io/fern/geoneric_dataset.h"
#include "fern/io/fern/utils.h"


namespace fern {

GeonericDriver::GeonericDriver()

    : Driver("Geoneric")

{
}


bool GeonericDriver::can_open_for_read(
    String const& name)
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


bool GeonericDriver::can_open_for_update(
    String const& name)
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


bool GeonericDriver::can_open_for_overwrite(
    String const& name)
{
    return (file_exists(name) && can_open_for_update(name)) ||
        directory_is_writable(Path(name).parent_path());
}


bool GeonericDriver::can_open(
    String const& name,
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


std::shared_ptr<Dataset> GeonericDriver::open(
    String const& name,
    OpenMode open_mode)
{
    return std::shared_ptr<Dataset>(new GeonericDataset(name, open_mode));
}

} // namespace fern
