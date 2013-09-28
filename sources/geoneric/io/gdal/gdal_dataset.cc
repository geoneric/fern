#include "geoneric/io/gdal/gdal_dataset.h"
#include "gdal_priv.h"
#include "geoneric/io/core/path.h"
#include "geoneric/feature/core/attributes.h"


namespace geoneric {

bool GDALDataset::can_open(
    String const& name)
{
    return GDALOpen(name.encode_in_default_encoding().c_str(), GA_ReadOnly) !=
        nullptr;
}


GDALDataset::GDALDataset(
    String const& name)

    : Dataset(name, OpenMode::READ)

{
    if(!exists()) {
        // TODO raise exception.
        assert(false);
    }
}


bool GDALDataset::exists() const
{
    // TODO
    return true;
}


size_t GDALDataset::nr_features() const
{
    return 1;
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
        // TODO raise exception
        assert(false);
    }

    String attribute_name = Path(this->name()).stem();
    assert(contains_attribute(attribute_name));

    std::shared_ptr<Feature> feature(new Feature());
    feature->add_attribute(attribute_name, std::dynamic_pointer_cast<Attribute>(
        read_attribute(attribute_name)));

    return feature;
}


std::shared_ptr<Attribute> GDALDataset::read_attribute(
    String const& name) const
{
    if(!contains_attribute(name)) {
        // TODO raise exception
        assert(false);
    }

    // Open file and read all cells.
    ::GDALDataset* dataset = static_cast<::GDALDataset*>(GDALOpen(
        this->name().encode_in_default_encoding().c_str(), GA_ReadOnly));

    if(!dataset) {
        // TODO raise exception
        assert(false);
    }

    int nr_rows = dataset->GetRasterYSize();
    int nr_cols = dataset->GetRasterXSize();

    double geo_transform[6];

    if(dataset->GetGeoTransform(geo_transform) != CE_None) {
        // TODO raise exception
        assert(false);
    }

    assert(geo_transform[1] == std::abs(geo_transform[5]));
    double cell_size = geo_transform[1];

    // Assume we need the first band only.
    assert(dataset->GetRasterCount() == 1);

    GDALRasterBand* band = dataset->GetRasterBand(1);
    assert(band);
    assert(band->GetRasterDataType() == GDT_Int32);

    d2::Point south_west;
    set<0>(south_west, geo_transform[0]);
    set<1>(south_west, geo_transform[3]);

    d2::Point north_east;
    set<0>(north_east, get<0>(south_west) + nr_cols * cell_size);
    set<1>(north_east, get<1>(south_west) + nr_rows * cell_size);

    d2::Box box(south_west, north_east);

    d1::ArrayValuePtr<int32_t> grid(new d1::ArrayValue<int32_t>(
        extents[nr_rows * nr_cols]));

    if(band->RasterIO(GF_Read, 0, 0, nr_cols, nr_rows, grid->data(), nr_cols,
            nr_rows, GDT_Int32, 0, 0) != CE_None) {
        // TODO raise exception
        assert(false);
    }

    FieldAttributePtr<int32_t> attribute(new FieldAttribute<int32_t>());
    FieldAttribute<int32_t>::GID gid = attribute->add(box, grid);

    return std::dynamic_pointer_cast<Attribute>(attribute);
}

} // namespace geoneric
