#include "fern/io/gdal/gdal_dataset.h"
#include "gdal_priv.h"
#include "fern/core/io_error.h"
#include "fern/core/path.h"
#include "fern/core/type_traits.h"
#include "fern/core/value_type_traits.h"
#include "fern/feature/visitor/attribute_type_visitor.h"
#include "fern/io/core/file.h"
#include "fern/io/gdal/gdal_data_type_traits.h"
#include "fern/io/gdal/gdal_type_traits.h"


namespace fern {
namespace {

::GDALDataset* gdal_open_for_read(
    GDALDriver const& driver,
    String const& name)
{
    GDALOpenInfo open_info(name.encode_in_default_encoding().c_str(),
        GA_ReadOnly);
    ::GDALDataset* dataset = static_cast<::GDALDataset*>(driver.pfnOpen(
        &open_info));

    if(!dataset) {
        if(!file_exists(name)) {
            throw IOError(name,
                Exception::messages()[MessageId::DOES_NOT_EXIST]);
        }
        else {
            throw IOError(name,
                Exception::messages()[MessageId::CANNOT_BE_READ]);
        }
    }

    return dataset;
}


::GDALDataset* gdal_open_for_update(
    GDALDriver const& driver,
    String const& name)
{
    GDALOpenInfo open_info(name.encode_in_default_encoding().c_str(),
        GA_Update);
    ::GDALDataset* dataset = static_cast<::GDALDataset*>(driver.pfnOpen(
        &open_info));

    if(!dataset) {
        if(!file_exists(name)) {
            throw IOError(name,
                Exception::messages()[MessageId::DOES_NOT_EXIST]);
        }
        else {
            throw IOError(name,
                Exception::messages()[MessageId::CANNOT_BE_WRITTEN]);
        }
    }

    return dataset;
}


::GDALDataset* gdal_open(
    GDALDriver const& driver,
    String const& name,
    OpenMode open_mode)
{
    ::GDALDataset* dataset = nullptr;

    switch(open_mode) {
        case OpenMode::READ: {
            dataset = gdal_open_for_read(driver, name);
            break;
        }
        case OpenMode::UPDATE: {
            dataset = gdal_open_for_update(driver, name);
            break;
        }
        case OpenMode::OVERWRITE: {
            // The dataset may not yet exist. In any case, we will be
            // overwriting it. If it exists, we can delete it now.
            driver.QuietDelete(name.encode_in_default_encoding().c_str());
            break;
        }
    }

    assert(dataset || open_mode == OpenMode::OVERWRITE);

    return dataset;
}


template<
    class T>
::GDALDataset* create_gdal_dataset(
    GDALDriver& driver,
    FieldAttribute<T> const& field,
    String const& name)
{
    assert(field.values().size() == 1u);
    FieldValue<T> const& array(*field.values().cbegin()->second);
    int nr_rows = array.shape()[0];
    int nr_cols = array.shape()[1];
    int nr_bands = 1;
    char** options = NULL;

    ::GDALDataset* dataset = driver.Create(
        name.encode_in_default_encoding().c_str(), nr_cols, nr_rows, nr_bands,
        GDALTypeTraits<T>::data_type, options);

    if(!dataset) {
        throw IOError(name,
            Exception::messages()[MessageId::CANNOT_BE_CREATED]);
    }

    return dataset;
}


#define GDAL_DATA_TYPE_TO_VALUE_TYPES_CASE(                                    \
        gdal_data_type)                                                        \
    case gdal_data_type: {                                                     \
        result = TypeTraits<GDALDataTypeTraits<                                \
            gdal_data_type>::type>::value_types;                               \
        break;                                                                 \
    }

ValueTypes gdal_data_type_to_value_types(
    GDALDataType data_type)
{
    ValueTypes result;

    switch(data_type) {
        GDAL_DATA_TYPE_TO_VALUE_TYPES_CASE(GDT_Byte)
        GDAL_DATA_TYPE_TO_VALUE_TYPES_CASE(GDT_UInt16)
        GDAL_DATA_TYPE_TO_VALUE_TYPES_CASE(GDT_Int16)
        GDAL_DATA_TYPE_TO_VALUE_TYPES_CASE(GDT_UInt32)
        GDAL_DATA_TYPE_TO_VALUE_TYPES_CASE(GDT_Int32)
        GDAL_DATA_TYPE_TO_VALUE_TYPES_CASE(GDT_Float32)
        GDAL_DATA_TYPE_TO_VALUE_TYPES_CASE(GDT_Float64)
        default: {
            assert(false);
            // TODO Throw error stating the data type is not supported. The
            //      caller should add more info.
        }
        // case GDT_CInt16: {
        //     throw IOError(this->name(),
        //         Exception::messages().format_message(
        //             MessageId::UNSUPPORTED_VALUE_TYPE,
        //             path, GDALDataTypeTraits<GDT_CInt16>::name));
        // }
        // case GDT_CInt32: {
        //     throw IOError(this->name(),
        //         Exception::messages().format_message(
        //             MessageId::UNSUPPORTED_VALUE_TYPE,
        //             path, GDALDataTypeTraits<GDT_CInt32>::name));
        // }
        // case GDT_CFloat32: {
        //     throw IOError(this->name(),
        //         Exception::messages().format_message(
        //             MessageId::UNSUPPORTED_VALUE_TYPE,
        //             path, GDALDataTypeTraits<GDT_CFloat32>::name));
        // }
        // case GDT_CFloat64: {
        //     throw IOError(this->name(),
        //         Exception::messages().format_message(
        //             MessageId::UNSUPPORTED_VALUE_TYPE,
        //             path, GDALDataTypeTraits<GDT_CFloat64>::name));
        // }
        // case GDT_TypeCount: {
        //     throw IOError(this->name(),
        //         Exception::messages().format_message(
        //             MessageId::UNSUPPORTED_VALUE_TYPE,
        //             path, GDALDataTypeTraits<GDT_TypeCount>::name));
        // }
        // case GDT_Unknown: {
        //     throw IOError(this->name(),
        //         Exception::messages().format_message(
        //             MessageId::UNSUPPORTED_VALUE_TYPE,
        //             path, GDALDataTypeTraits<GDT_Unknown>::name));
        // }
    }

    return result;
}

#undef GDAL_DATA_TYPE_TO_VALUE_TYPES_CASE

} // Anonymous namespace


GDALDataset::GDALDataset(
    ::GDALDriver* driver,
    String const& name,
    OpenMode open_mode)

    : Dataset(name, open_mode),
      _driver(driver),
      _dataset(nullptr)

{
    assert(driver);
    _dataset = gdal_open(*_driver, this->name(), this->open_mode());
}


GDALDataset::GDALDataset(
    String const& format,
    String const& name,
    OpenMode open_mode)

    : Dataset(name, open_mode),
      _driver(GetGDALDriverManager()->GetDriverByName(
          format.encode_in_utf8().c_str())),
      _dataset(nullptr)

{
    assert(_driver);
    _dataset = gdal_open(*_driver, this->name(), this->open_mode());
}


GDALDataset::GDALDataset(
    ::GDALDataset* dataset,
    String const& name,
    OpenMode open_mode)

    : Dataset(name, open_mode),
      _driver(dataset->GetDriver()),
      _dataset(dataset)

{
}


GDALDataset::~GDALDataset()
{
    // If this instance is created with open mode OVERWRITE, and the dataset
    // has not been created, then _dataset is still nullptr.
    if(_dataset) {
        GDALClose(_dataset);
    }
}


size_t GDALDataset::nr_features() const
{
    return 1u;
}


std::vector<String> GDALDataset::feature_names() const
{
    return std::vector<String>{Path(this->name()).stem()};
}


bool GDALDataset::contains_feature(
    Path const& path) const
{
    // GDAL raster datasets contain one root feature, but no sub-features.
    // /<raster>
    return path == String("/") + String(Path(this->name()).stem());
}


bool GDALDataset::contains_attribute(
    Path const& path) const
{
    // The name of the one attribute in a GDAL raster equals the name of the
    // dataset without leading path and extension. It is prefixed by the path
    // to the feature, which is also named after the raster.
    // /<raster>/<raster>
    return Path("/" + String(Path(this->name()).stem()) + "/" +
        String(Path(this->name()).stem())) == path;
}


ExpressionType GDALDataset::expression_type(
    Path const& path) const
{
    assert(_dataset);

    if(!contains_attribute(path)) {
        throw IOError(this->name(),
            Exception::messages().format_message(
                MessageId::DOES_NOT_CONTAIN_FEATURE, path));
    }

    ExpressionType result;

    // Assume we need the first band only.
    assert(_dataset->GetRasterCount() == 1);
    GDALRasterBand* band = _dataset->GetRasterBand(1);
    assert(band);
    result = ExpressionType(DataTypes::STATIC_FIELD,
        gdal_data_type_to_value_types(band->GetRasterDataType()));

    return result;
}


std::shared_ptr<Feature> GDALDataset::open_feature(
    Path const& /* path */) const
{
    std::shared_ptr<Feature> result;
    // TODO
    assert(false);
    return result;
}


template<
    class T>
std::shared_ptr<FieldAttribute<T>> GDALDataset::open_attribute(
    GDALRasterBand& /* band */) const
{
    FieldAttributePtr<T> attribute(new FieldAttribute<T>());
    return attribute;
}


#define OPEN_CASE(                                                             \
        value_type)                                                            \
    case value_type: {                                                         \
        result = open_attribute<ValueTypeTraits<value_type>::type>(*band);     \
        break;                                                                 \
    }

std::shared_ptr<Attribute> GDALDataset::open_attribute(
    Path const& path) const
{
    GDALRasterBand* band = this->band(path);
    ValueType value_type = this->value_type(*band, path);

    std::shared_ptr<Attribute> result;
    switch(value_type) {
        OPEN_CASE(VT_UINT8);
        OPEN_CASE(VT_UINT16);
        OPEN_CASE(VT_UINT32);
        OPEN_CASE(VT_INT16);
        OPEN_CASE(VT_INT32);
        OPEN_CASE(VT_FLOAT32);
        OPEN_CASE(VT_FLOAT64);
        case VT_STRING:
        case VT_UINT64:
        case VT_INT8:
        case VT_INT64: {
            // These aren't support by gdal, so this shouldn't happen.
            assert(false);
            break;
        }
    }

    assert(result);
    return result;
}

#undef OPEN_CASE


std::shared_ptr<Feature> GDALDataset::read_feature(
    Path const& path) const
{
    if(!contains_feature(path)) {
        throw IOError(this->name(),
            Exception::messages().format_message(
                MessageId::DOES_NOT_CONTAIN_FEATURE, path));
    }

    Path attribute_path = path + "/" + Path(this->name()).stem();
    assert(contains_attribute(attribute_path));

    std::shared_ptr<Feature> feature(new Feature());
    feature->add_attribute(String(attribute_path.filename()),
        std::dynamic_pointer_cast<Attribute>(read_attribute(attribute_path)));

    return feature;
}


template<
    class T>
std::shared_ptr<Attribute> GDALDataset::read_attribute(
    GDALRasterBand& band) const
{
    int const nr_rows = _dataset->GetRasterYSize();
    int const nr_cols = _dataset->GetRasterXSize();

    double geo_transform[6];

    if(_dataset->GetGeoTransform(geo_transform) != CE_None) {
        // According to the gdal docs:
        // The default transform is (0,1,0,0,0,1) and should be returned even
        // when a CE_Failure error is returned, such as for formats that don't
        // support transformation to projection coordinates.
        geo_transform[0] = 0.0;
        geo_transform[1] = 1.0;
        geo_transform[2] = 0.0;
        geo_transform[3] = 0.0;
        geo_transform[4] = 0.0;
        geo_transform[5] = 1.0;
    }

    assert(geo_transform[1] == std::abs(geo_transform[5]));
    double const cell_size = geo_transform[1];

    d2::Point south_west, north_east;
    set<0>(south_west, geo_transform[0]);
    set<1>(north_east, geo_transform[3]);
    set<0>(north_east, get<0>(south_west) + nr_cols * cell_size);
    set<1>(south_west, get<1>(north_east) - nr_rows * cell_size);

    d2::Box box(south_west, north_east);

    FieldValuePtr<T> array(new FieldValue<T>(extents[nr_rows][nr_cols]));

    assert(band.GetRasterDataType() == GDALTypeTraits<T>::data_type);
    if(band.RasterIO(GF_Read, 0, 0, nr_cols, nr_rows, array->data(), nr_cols,
            nr_rows, GDALTypeTraits<T>::data_type, 0, 0) != CE_None) {
        // This shouldn't happen.
        throw IOError(this->name(),
            Exception::messages()[MessageId::UNKNOWN_ERROR]);
    }

    // http://trac.osgeo.org/gdal/wiki/rfc15_nodatabitmask
    int mask_flags = band.GetMaskFlags();
    if(!(mask_flags & GMF_ALL_VALID)) {
        assert(!(mask_flags & GMF_ALPHA));
        GDALRasterBand* mask_band = band.GetMaskBand();
        assert(mask_band->GetRasterDataType() == GDT_Byte);
        // The mask band has gdal data type GDT_Byte. A value of zero
        // means that the value must be masked.
        ArrayValue<typename GDALDataTypeTraits<GDT_Byte>::type, 2> mask(
            extents[nr_rows][nr_cols]);

        if(mask_band->RasterIO(GF_Read, 0, 0, nr_cols, nr_rows, mask.data(),
                nr_cols, nr_rows, GDT_Byte, 0, 0) != CE_None) {
            // This shouldn't happen.
            throw IOError(this->name(),
                Exception::messages()[MessageId::UNKNOWN_ERROR]);
        }

        array->set_mask(mask);
    }

    std::shared_ptr<FieldAttribute<T>> attribute(open_attribute<T>(band));
    typename FieldAttribute<T>::GID gid = attribute->add(box, array);

    return std::dynamic_pointer_cast<Attribute>(attribute);
}


GDALRasterBand* GDALDataset::band(
    Path const& path) const
{
    assert(_dataset);

    if(!contains_attribute(path)) {
        throw IOError(this->name(),
            Exception::messages().format_message(
                MessageId::DOES_NOT_CONTAIN_ATTRIBUTE, path));
    }

    // Assume we need the first band only.
    assert(_dataset->GetRasterCount() == 1);

    GDALRasterBand* band = _dataset->GetRasterBand(1);
    assert(band);
    return band;
}


#define RASTER_DATA_TYPE_CASE(                                                 \
        data_type)                                                             \
    case data_type: {                                                          \
        result = TypeTraits<GDALDataTypeTraits<data_type>::type>::value_type;  \
        break;                                                                 \
    }

// TODO exception
#define UNSUPPORTED_RASTER_DATA_TYPE_CASE(                                     \
        data_type)                                                             \
    case data_type: {                                                          \
        throw IOError(this->name(),                                            \
            Exception::messages().format_message(                              \
                MessageId::UNSUPPORTED_VALUE_TYPE,                             \
                path, GDALDataTypeTraits<data_type>::name));                   \
        break;                                                                 \
    }

ValueType GDALDataset::value_type(
    GDALRasterBand& band,
    Path const& path) const
{
    ValueType result;

    switch(band.GetRasterDataType()) {
        RASTER_DATA_TYPE_CASE(GDT_Byte);
        RASTER_DATA_TYPE_CASE(GDT_UInt16);
        RASTER_DATA_TYPE_CASE(GDT_Int16);
        RASTER_DATA_TYPE_CASE(GDT_UInt32);
        RASTER_DATA_TYPE_CASE(GDT_Int32);
        RASTER_DATA_TYPE_CASE(GDT_Float32);
        RASTER_DATA_TYPE_CASE(GDT_Float64);
        UNSUPPORTED_RASTER_DATA_TYPE_CASE(GDT_CInt16);
        UNSUPPORTED_RASTER_DATA_TYPE_CASE(GDT_CInt32);
        UNSUPPORTED_RASTER_DATA_TYPE_CASE(GDT_CFloat32);
        UNSUPPORTED_RASTER_DATA_TYPE_CASE(GDT_CFloat64);
        UNSUPPORTED_RASTER_DATA_TYPE_CASE(GDT_TypeCount);
        UNSUPPORTED_RASTER_DATA_TYPE_CASE(GDT_Unknown);
    }

    return result;
}

#undef UNSUPPORTED_RASTER_DATA_TYPE_CASE
#undef RASTER_DATA_TYPE_CASE


#define READ_CASE(                                                             \
        value_type)                                                            \
    case value_type: {                                                         \
        result = read_attribute<ValueTypeTraits<value_type>::type>(*band);     \
        break;                                                                 \
    }

std::shared_ptr<Attribute> GDALDataset::read_attribute(
    Path const& path) const
{
    GDALRasterBand* band = this->band(path);
    ValueType value_type = this->value_type(*band, path);

    std::shared_ptr<Attribute> result;
    switch(value_type) {
        READ_CASE(VT_UINT8);
        READ_CASE(VT_UINT16);
        READ_CASE(VT_UINT32);
        READ_CASE(VT_INT16);
        READ_CASE(VT_INT32);
        READ_CASE(VT_FLOAT32);
        READ_CASE(VT_FLOAT64);
        case VT_STRING:
        case VT_UINT64:
        case VT_INT8:
        case VT_INT64: {
            // These aren't support by gdal, so this shouldn't happen.
            assert(false);
            break;
        }
    }

    assert(result);
    return result;
}


template<
    class T>
void GDALDataset::write_attribute(
    FieldAttribute<T> const& field,
    Path const& path)
{
    assert((_dataset && (open_mode() == OpenMode::UPDATE)) ||
        (!_dataset && (open_mode() == OpenMode::OVERWRITE)));

    if(!_dataset && open_mode() == OpenMode::OVERWRITE) {
        // Upon creation of this instance, the dataset, if it existed, was
        // deleted. Now create a new dataset.
        _dataset = create_gdal_dataset(*_driver, field, this->name());
    }

    assert(contains_feature(path.parent_path()));
    assert(contains_attribute(path));
    assert(_dataset);
    assert(_dataset->GetRasterCount() == 1);

    assert(field.values().size() == 1u);
    FieldValue<T> const& array(*field.values().cbegin()->second);
    int nr_rows = array.shape()[0];
    int nr_cols = array.shape()[1];

    double geo_transform[6];
    {
        FieldDomain const& domain(field.domain());
        assert(domain.size() == 1u);
        d2::Box const& box(domain.cbegin()->second);
        geo_transform[0] = get<0>(box.min_corner());  // west
        geo_transform[1] =
            (get<0>(box.max_corner()) - get<0>(box.min_corner())) / nr_cols;
        geo_transform[2] = 0.0;
        geo_transform[3] = get<1>(box.max_corner());  // north
        geo_transform[4] = 0.0;
        geo_transform[5] =
            (get<1>(box.max_corner()) - get<1>(box.min_corner())) / nr_rows;
        assert(geo_transform[1] > 0.0);
        assert(geo_transform[1] == geo_transform[5]);
    }

    if(_dataset->SetGeoTransform(geo_transform) != CE_None) {
        throw IOError(this->name(),
            Exception::messages()[MessageId::CANNOT_BE_WRITTEN]);
    }

    GDALRasterBand* band = _dataset->GetRasterBand(1);
    assert(band);

    if(band->RasterIO(GF_Write, 0, 0, nr_cols, nr_rows, const_cast<T*>(
            array.data()), nr_cols, nr_rows, GDALTypeTraits<T>::data_type,
            0, 0) != CE_None) {
        throw IOError(this->name(),
            Exception::messages()[MessageId::CANNOT_BE_WRITTEN]);
    }

    if(array.has_masked_values()) {
        // This assumes the dataset contains onle one band. The mask is taken to
        // be global to the dataset.
        if(band->CreateMaskBand(GMF_PER_DATASET) != CE_None) {
            throw IOError(this->name(),
                Exception::messages()[MessageId::CANNOT_BE_WRITTEN]);
        }

        GDALRasterBand* mask_band = band->GetMaskBand();
        assert(mask_band->GetRasterDataType() == GDT_Byte);
        // The mask band has gdal data type GDT_Byte. A value of zero
        // means that the value must be masked.
        ArrayValue<typename GDALDataTypeTraits<GDT_Byte>::type, 2> mask(
            extents[nr_rows][nr_cols]);
        array.mask(mask);

        if(mask_band->RasterIO(GF_Write, 0, 0, nr_cols, nr_rows, mask.data(),
                nr_cols, nr_rows, GDT_Byte, 0, 0) != CE_None) {
            throw IOError(this->name(),
                Exception::messages()[MessageId::CANNOT_BE_WRITTEN]);
        }
    }
}


#define WRITE_ATTRIBUTE_CASE(                                                  \
        value_type)                                                            \
case value_type: {                                                             \
    write_attribute(                                                           \
        dynamic_cast<FieldAttribute<                                           \
            ValueTypeTraits<value_type>::type> const&>(attribute), path);      \
    break;                                                                     \
}

void GDALDataset::write_attribute(
    Attribute const& attribute,
    Path const& path)
{
    AttributeTypeVisitor visitor;
    attribute.Accept(visitor);

    switch(visitor.data_type()) {
        case DT_CONSTANT: {
            assert(false); // Data type not supported by driver.
            break;
        }
        case DT_STATIC_FIELD: {
            switch(visitor.value_type()) {
                WRITE_ATTRIBUTE_CASE(VT_UINT8)
                WRITE_ATTRIBUTE_CASE(VT_UINT16)
                WRITE_ATTRIBUTE_CASE(VT_INT16)
                WRITE_ATTRIBUTE_CASE(VT_UINT32)
                WRITE_ATTRIBUTE_CASE(VT_INT32)
                WRITE_ATTRIBUTE_CASE(VT_FLOAT32)
                WRITE_ATTRIBUTE_CASE(VT_FLOAT64)
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
}

#undef WRITE_ATTRIBUTE_CASE

} // namespace fern
