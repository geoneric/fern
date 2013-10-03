#include "geoneric/io/gdal/gdal_dataset.h"
#include "gdal_priv.h"
#include "geoneric/core/io_error.h"
#include "geoneric/io/core/file.h"
#include "geoneric/io/core/path.h"
#include "geoneric/feature/core/attributes.h"
#include "geoneric/io/gdal/gdal_data_type_traits.h"
#include "geoneric/io/gdal/gdal_type_traits.h"


namespace geoneric {

bool GDALDataset::can_open(
    String const& name)
{
    // can_open_for_read, can_read!
    return GDALOpen(name.encode_in_default_encoding().c_str(), GA_ReadOnly) !=
        nullptr;
}


GDALDataset::GDALDataset(
    String const& name)

    : Dataset(name, OpenMode::READ),
      _dataset(static_cast<::GDALDataset*>(GDALOpen(
          this->name().encode_in_default_encoding().c_str(), GA_ReadOnly)))

{
    if(!_dataset) {
        if(!file_exists(name)) {
            throw IOError(name,
                Exception::messages()[MessageId::DOES_NOT_EXIST]);
        }
        else {
            throw IOError(name,
                Exception::messages()[MessageId::CANNOT_BE_READ]);
        }
    }
}


GDALDataset::~GDALDataset()
{
    assert(_dataset);
    GDALClose(_dataset);
}


size_t GDALDataset::nr_features() const
{
    return 1u;
}


bool GDALDataset::contains_feature(
    String const& name) const
{
    // GDAL raster datasets contain the root feature, but no sub-features.
    return name == "/";
}


bool GDALDataset::contains_attribute(
    String const& name) const
{
    // The name of the one attribute in a GDAL raster equals the name of the
    // dataset without leading path and extension.
    return String(Path(this->name()).stem()) == name;
}


std::shared_ptr<Feature> GDALDataset::read_feature(
    String const& name) const
{
    if(!contains_feature(name)) {
        throw IOError(this->name(),
            Exception::messages().format_message(
                MessageId::DOES_NOT_CONTAIN_FEATURE, name));
    }

    String attribute_name = Path(this->name()).stem();
    assert(contains_attribute(attribute_name));

    std::shared_ptr<Feature> feature(new Feature());
    feature->add_attribute(attribute_name, std::dynamic_pointer_cast<Attribute>(
        read_attribute(attribute_name)));

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
        // This shouldn't happen.
        throw IOError(this->name(),
            Exception::messages()[MessageId::UNKNOWN_ERROR]);
    }

    assert(geo_transform[1] == std::abs(geo_transform[5]));
    double const cell_size = geo_transform[1];

    d2::Point south_west;
    set<0>(south_west, geo_transform[0]);
    set<1>(south_west, geo_transform[3]);

    d2::Point north_east;
    set<0>(north_east, get<0>(south_west) + nr_cols * cell_size);
    set<1>(north_east, get<1>(south_west) + nr_rows * cell_size);

    d2::Box box(south_west, north_east);

    FieldValuePtr<T> grid(new FieldValue<T>(extents[nr_rows][nr_cols]));

    assert(band.GetRasterDataType() == GDALTypeTraits<T>::data_type);
    if(band.RasterIO(GF_Read, 0, 0, nr_cols, nr_rows, grid->data(), nr_cols,
            nr_rows, GDALTypeTraits<T>::data_type, 0, 0) != CE_None) {
        // This shouldn't happen.
        throw IOError(this->name(),
            Exception::messages()[MessageId::UNKNOWN_ERROR]);
    }

    int success = 0;
    T nodata_value = static_cast<T>(band.GetNoDataValue(&success));

    if(success) {
        grid->mask_value(nodata_value);
    }

    FieldAttributePtr<T> attribute(new FieldAttribute<T>());
    typename FieldAttribute<T>::GID gid = attribute->add(box, grid);

    return std::dynamic_pointer_cast<Attribute>(attribute);
}


std::shared_ptr<Attribute> GDALDataset::read_attribute(
    String const& name) const
{
    assert(_dataset);

    if(!contains_attribute(name)) {
        throw IOError(this->name(),
            Exception::messages().format_message(
                MessageId::DOES_NOT_CONTAIN_ATTRIBUTE, name));
    }

    // Assume we need the first band only.
    assert(_dataset->GetRasterCount() == 1);

    GDALRasterBand* band = _dataset->GetRasterBand(1);
    assert(band);

    std::shared_ptr<Attribute> result;

    switch(band->GetRasterDataType()) {
        case GDT_Byte: {
            result = read_attribute<uint8_t>(*band);
            break;
        }
        case GDT_UInt16: {
            result = read_attribute<uint16_t>(*band);
            break;
        }
        case GDT_Int16: {
            result = read_attribute<int16_t>(*band);
            break;
        }
        case GDT_UInt32: {
            result = read_attribute<uint32_t>(*band);
            break;
        }
        case GDT_Int32: {
            result = read_attribute<int32_t>(*band);
            break;
        }
        case GDT_Float32: {
            result = read_attribute<float>(*band);
            break;
        }
        case GDT_Float64: {
            result = read_attribute<double>(*band);
            break;
        }
        case GDT_CInt16: {
            throw IOError(this->name(),
                Exception::messages().format_message(
                    MessageId::UNSUPPORTED_VALUE_TYPE,
                    name, GDALDataTypeTraits<GDT_CInt16>::name));
        }
        case GDT_CInt32: {
            throw IOError(this->name(),
                Exception::messages().format_message(
                    MessageId::UNSUPPORTED_VALUE_TYPE,
                    name, GDALDataTypeTraits<GDT_CInt32>::name));
        }
        case GDT_CFloat32: {
            throw IOError(this->name(),
                Exception::messages().format_message(
                    MessageId::UNSUPPORTED_VALUE_TYPE,
                    name, GDALDataTypeTraits<GDT_CFloat32>::name));
        }
        case GDT_CFloat64: {
            throw IOError(this->name(),
                Exception::messages().format_message(
                    MessageId::UNSUPPORTED_VALUE_TYPE,
                    name, GDALDataTypeTraits<GDT_CFloat64>::name));
        }
        case GDT_TypeCount: {
            throw IOError(this->name(),
                Exception::messages().format_message(
                    MessageId::UNSUPPORTED_VALUE_TYPE,
                    name, GDALDataTypeTraits<GDT_TypeCount>::name));
        }
        case GDT_Unknown: {
            throw IOError(this->name(),
                Exception::messages().format_message(
                    MessageId::UNSUPPORTED_VALUE_TYPE,
                    name, GDALDataTypeTraits<GDT_Unknown>::name));
        }
    }

    assert(result);
    return result;
}

} // namespace geoneric
