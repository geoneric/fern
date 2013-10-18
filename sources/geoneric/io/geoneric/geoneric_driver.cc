#include "geoneric/io/geoneric/geoneric_driver.h"
#include "geoneric/core/data_name.h"
#include "geoneric/feature/visitor/attribute_type_visitor.h"
#include "geoneric/io/geoneric/geoneric_dataset.h"
#include "geoneric/io/geoneric/utils.h"


namespace geoneric {

GeonericDriver::GeonericDriver()

    : Driver()

{
}


bool GeonericDriver::can_open_for_read(
    String const& /* name */)
{
    // GDALOpenInfo open_info(name.encode_in_default_encoding().c_str(),
    //     GA_ReadOnly);
    // return _driver->pfnOpen(&open_info) != nullptr;


    // bool result = false;

    // try {
    //     result = H5::H5File::isHdf5(name.encode_in_default_encoding().c_str());
    // }
    // catch(H5::FileIException const& /* exception */) {
    //     result = false;
    // }

    // return result;

    return false;
}


bool GeonericDriver::can_open_for_update(
    String const& /* name */)
{
    // GDALOpenInfo open_info(name.encode_in_default_encoding().c_str(),
    //     GA_Update);
    // return _driver->pfnOpen(&open_info) != nullptr;

    return false;
}


bool GeonericDriver::can_open(
    String const& name,
    OpenMode open_mode)
{
    return open_mode == OpenMode::READ
        ? can_open_for_read(name)
        : can_open_for_update(name)
        ;
}


bool GeonericDriver::exists(
    String const& name,
    OpenMode open_mode)
{
    return can_open(name, open_mode);
}


std::shared_ptr<Dataset> GeonericDriver::open(
    String const& /* name */,
    OpenMode /* open_mode */)
{
    std::shared_ptr<Dataset> result;

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
