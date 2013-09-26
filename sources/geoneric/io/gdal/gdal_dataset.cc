#include "geoneric/io/gdal/gdal_dataset.h"
#include "gdal_priv.h"
#include "geoneric/io/core/path.h"
#include "geoneric/feature/core/array_value.h"
#include "geoneric/feature/core/box.h"
#include "geoneric/feature/core/point.h"
#include "geoneric/feature/core/spatial_attribute.h"
#include "geoneric/feature/core/spatial_domain.h"


namespace geoneric {

GDALDataset::GDALDataset(
    String const& name)

    : Dataset(name, OpenMode::READ)

{
    // TODO Move somewhere else.
    GDALAllRegister();

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
    return Path(this->name()).stem() == name;
}


std::shared_ptr<Feature> GDALDataset::read(
    String const& name) const
{
    if(!contains_feature(name)) {
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

    typedef Point<double, 2> Point;
    typedef Box<Point> Box;
    typedef SpatialDomain<Box> BoxDomain;
    typedef ArrayValue<int32_t, 1> Value;
    typedef std::shared_ptr<Value> ValuePtr;
    typedef SpatialAttribute<BoxDomain, ValuePtr> BoxesAttribute;
    typedef std::shared_ptr<BoxesAttribute> BoxesAttributePtr;

    Point south_west;
    set<0>(south_west, geo_transform[0]);
    set<1>(south_west, geo_transform[3]);

    Point north_east;
    set<0>(north_east, get<0>(south_west) + nr_cols * cell_size);
    set<1>(north_east, get<1>(south_west) + nr_rows * cell_size);

    Box box(south_west, north_east);

    ValuePtr grid(new Value(extents[nr_rows * nr_cols]));

    if(band->RasterIO(GF_Read, 0, 0, nr_cols, nr_rows, grid->data(), nr_cols,
            nr_rows, GDT_Int32, 0, 0) != CE_None) {
        // TODO raise exception
        assert(false);
    }

    BoxesAttributePtr attribute(new BoxesAttribute());
    BoxesAttribute::GID gid = attribute->add(box, grid);

    std::shared_ptr<Feature> feature(new Feature());
    (*feature)[name] = std::dynamic_pointer_cast<Attribute>(attribute);

    return feature;
}

} // namespace geoneric
