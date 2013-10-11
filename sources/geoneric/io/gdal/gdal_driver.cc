#include "geoneric/io/gdal/gdal_driver.h"
#include "gdal_priv.h"
#include "geoneric/core/io_error.h"
#include "geoneric/core/value_type_traits.h"
#include "geoneric/feature/visitor/attribute_type_visitor.h"
#include "geoneric/io/gdal/gdal_type_traits.h"
#include "geoneric/io/gdal/gdal_dataset.h"


namespace geoneric {

GDALDriver::GDALDriver(
    String const& format)

    : Driver(),
      _format(format),
      _driver(GetGDALDriverManager()->GetDriverByName(
          format.encode_in_default_encoding().c_str()))

{
    if(!_driver) {
        // TODO Driver not available.
        std::cout << "format: " << format << std::endl;
        assert(false);
    }
}


GDALDriver::GDALDriver(
    ::GDALDriver* driver)

    : Driver(),
      _format(driver->GetDescription()),
      _driver(driver)

{
}


bool GDALDriver::can_open(
    String const& name,
    OpenMode open_mode)
{
    return open_mode == OpenMode::READ
        ? can_open_for_read(name)
        : can_open_for_update(name)
        ;
}


bool GDALDriver::can_open_for_read(
    String const& name)
{
    GDALOpenInfo open_info(name.encode_in_default_encoding().c_str(),
        GA_ReadOnly);
    return _driver->pfnOpen(&open_info) != nullptr;
}


bool GDALDriver::can_open_for_update(
    String const& name)
{
    GDALOpenInfo open_info(name.encode_in_default_encoding().c_str(),
        GA_Update);
    return _driver->pfnOpen(&open_info) != nullptr;
}


bool GDALDriver::exists(
    String const& name,
    OpenMode open_mode)
{
    return can_open(name, open_mode);
}


std::shared_ptr<Dataset> GDALDriver::open(
    String const& name,
    OpenMode open_mode)
{
    std::shared_ptr<Dataset> result;

    if(can_open(name, open_mode)) {
        result = std::shared_ptr<Dataset>(new GDALDataset(_driver, name,
            open_mode));
    }

    return result;
}


template<
    class T>
std::shared_ptr<Dataset> GDALDriver::create(
    FieldAttribute<T> const& field,
    String const& name)
{
    assert(field.values().size() == 1u);
    FieldValue<T> const& array(*field.values().cbegin()->second);
    int nr_rows = array.shape()[0];
    int nr_cols = array.shape()[1];
    int nr_bands = 1;
    char** options = NULL;

    ::GDALDataset* dataset = _driver->Create(
        name.encode_in_default_encoding().c_str(), nr_cols, nr_rows, nr_bands,
        GDALTypeTraits<T>::data_type, options);

    if(!dataset) {
        throw IOError(name,
            Exception::messages()[MessageId::CANNOT_BE_CREATED]);
    }

    return std::shared_ptr<Dataset>(new GDALDataset(dataset, name,
        OpenMode::OVERWRITE));
}


#define CREATE_CASE(                                                           \
        value_type)                                                            \
case value_type: {                                                             \
    result = create(dynamic_cast<FieldAttribute<                               \
        ValueTypeTraits<value_type>::type> const&>(attribute), name);          \
    break;                                                                     \
}

std::shared_ptr<Dataset> GDALDriver::create(
    Attribute const& attribute,
    String const& name)
{
    // Quiet delete is OK here.
    assert(_driver);
    _driver->QuietDelete(name.encode_in_default_encoding().c_str());

    AttributeTypeVisitor visitor;
    attribute.Accept(visitor);
    std::shared_ptr<Dataset> result;

    switch(visitor.data_type()) {
        case DT_CONSTANT: {
            assert(false); // Data type not supported by driver.
            break;
        }
        case DT_STATIC_FIELD: {
            switch(visitor.value_type()) {
                CREATE_CASE(VT_UINT8)
                CREATE_CASE(VT_UINT16)
                CREATE_CASE(VT_INT16)
                CREATE_CASE(VT_UINT32)
                CREATE_CASE(VT_INT32)
                CREATE_CASE(VT_FLOAT32)
                CREATE_CASE(VT_FLOAT64)
                case VT_INT8:
                case VT_UINT64:
                case VT_INT64:
                case VT_STRING: {
                    assert(false); // Value type not supported by driver.
                    break;
                }
            }
            break;
        }
    }

    assert(result);
    return result;
}

#undef CREATE_CASE

} // namespace geoneric
