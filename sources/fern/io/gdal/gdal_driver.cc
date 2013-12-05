#include "fern/io/gdal/gdal_driver.h"
#include "gdal_priv.h"
#include "fern/core/io_error.h"
#include "fern/core/value_type_traits.h"
#include "fern/feature/visitor/attribute_type_visitor.h"
#include "fern/io/core/file.h"
#include "fern/io/gdal/gdal_type_traits.h"
#include "fern/io/gdal/gdal_dataset.h"


namespace fern {

static auto deleter = [](::GDALDataset* dataset)
    { if(dataset) { GDALClose(dataset); } };
typedef std::unique_ptr<::GDALDataset, decltype(deleter)> GDALDatasetPtr;


GDALDriver::GDALDriver(
    String const& name)

    : Driver(name),
      _driver(GetGDALDriverManager()->GetDriverByName(
          name.encode_in_default_encoding().c_str()))

{
    if(!_driver) {
        // TODO Driver not available.
        std::cout << "name: " << name << std::endl;
        assert(false);
    }
}


GDALDriver::GDALDriver(
    ::GDALDriver* driver)

    : Driver(driver->GetDescription()),
      _driver(driver)

{
}


bool GDALDriver::can_open(
    String const& name,
    OpenMode open_mode)
{
    bool result = false;

    switch(open_mode) {
        case OpenMode::READ: {
          result = can_open_for_read(name);
          break;
        }
        case OpenMode::UPDATE: {
          result = can_open_for_update(name);
          break;
        }
        case OpenMode::OVERWRITE: {
          result = can_open_for_overwrite(name);
          break;
        }
    }

    return result;
}


bool GDALDriver::can_open_for_read(
    String const& name)
{
    GDALOpenInfo open_info(name.encode_in_default_encoding().c_str(),
        GA_ReadOnly);
    GDALDatasetPtr dataset(_driver->pfnOpen(&open_info), deleter);
    return static_cast<bool>(dataset);
}


bool GDALDriver::can_open_for_update(
    String const& name)
{
    GDALOpenInfo open_info(name.encode_in_default_encoding().c_str(),
        GA_Update);
    GDALDatasetPtr dataset(_driver->pfnOpen(&open_info), deleter);
    return static_cast<bool>(dataset);
}


bool GDALDriver::can_open_for_overwrite(
    String const& name)
{
    // This assumes the driver manages file-based datasets.
    return (file_exists(name) && can_open_for_update(name)) ||
        directory_is_writable(Path(name).parent_path());
}


// ExpressionType GDALDriver::expression_type(
//     DataName const& data_name)
// {
//     return open(data_name.database_pathname(),
//         OpenMode::READ)->expression_type(data_name.data_pathname());
// }


std::shared_ptr<Dataset> GDALDriver::open(
    String const& name,
    OpenMode open_mode)
{
    return std::shared_ptr<Dataset>(std::make_shared<GDALDataset>(_driver,
        name, open_mode));
}

} // namespace fern
