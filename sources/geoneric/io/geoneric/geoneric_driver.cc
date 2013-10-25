#include "geoneric/io/geoneric/geoneric_driver.h"
#include "geoneric/core/data_name.h"
#include "geoneric/feature/visitor/attribute_type_visitor.h"
#include "geoneric/io/geoneric/geoneric_dataset.h"
#include "geoneric/io/geoneric/utils.h"


namespace geoneric {

GeonericDriver::GeonericDriver()

    : Driver("Geoneric")

{
}


bool GeonericDriver::can_open_for_read(
    String const& name)
{
    // bool result = false;

    // try {
    //     result = H5::H5File::isHdf5(name.encode_in_default_encoding().c_str());
    // }
    // catch(H5::FileIException const& /* exception */) {
    //     result = false;
    // }

    // return result;

    bool result = false;

    try {
        open_file(name, OpenMode::READ);
        result = true;
    }
    catch(...) {
    }

    return result;
}


bool GeonericDriver::can_open_for_overwrite(
    String const& name)
{
    // Don't use OVERWRITE here. It will create a new file.
    // return static_cast<bool>(open_file(name, OpenMode::UPDATE));

    return can_open_for_update(name);
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


bool GeonericDriver::can_open(
    String const& name,
    OpenMode open_mode)
{
    // return static_cast<bool>(open_file(name, open_mode));
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


bool GeonericDriver::exists(
    String const& name,
    OpenMode open_mode)
{
    return can_open(name, open_mode);
}


std::shared_ptr<Dataset> GeonericDriver::open(
    String const& name,
    OpenMode open_mode)
{
    std::shared_ptr<Dataset> result;

    if(can_open(name, open_mode)) {
        result = std::shared_ptr<Dataset>(new GeonericDataset(name, open_mode));
    }

    return result;
}


std::shared_ptr<Dataset> GeonericDriver::create(
    Attribute const& attribute,
    String const& name)
{
    AttributeTypeVisitor visitor;
    attribute.Accept(visitor);

    // TODO Check if we support this attribute's type.

    DataName data_name(name);
    std::shared_ptr<H5::H5File> file(open_file(data_name.database_pathname(),
        OpenMode::OVERWRITE));
    std::shared_ptr<Dataset> result;

    if(file) {
        result.reset(new GeonericDataset(file, data_name.data_pathname(),
            OpenMode::OVERWRITE));
    }

    // TODO Write attribute.

    return result;
}

} // namespace geoneric
