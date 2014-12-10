#include <boost/python.hpp>
#include "fern/python_extension/core/init_python_module.h"
#include "fern/python_extension/feature/masked_raster.h"
#include "fern/python_extension/feature/numpy.h"


namespace bp = boost::python;
namespace fp = fern::python;


BOOST_PYTHON_MODULE(_fern_feature)
{
    INIT_PYTHON_MODULE("C++ module with wrappers for C++ types.")

    bp::numeric::array::set_module_and_type("numpy", "ndarray");

    bp::enum_<fern::ValueType>(
        "ValueType",
        "Type ids for all supported value types")
        .value("int8", fern::VT_INT8)
        .value("uint8", fern::VT_UINT8)
        .value("int16", fern::VT_INT16)
        .value("uint16", fern::VT_UINT16)
        .value("int32", fern::VT_INT32)
        .value("uint32", fern::VT_UINT32)
        .value("int64", fern::VT_INT64)
        .value("uint64", fern::VT_UINT64)
        .value("float32", fern::VT_FLOAT32)
        .value("float32", fern::VT_FLOAT32)
        .value("float64", fern::VT_FLOAT64)
        .value("float64", fern::VT_FLOAT64)
        .export_values()
        ;

    bp::class_<fp::MaskedRaster, fp::MaskedRasterHandle, boost::noncopyable>(
        "MaskedRaster",
        "Class for masked rasters."
        "\n"
        "A masked raster is a raster for which some cells may be masked "
        "out",
        bp::init<
            bp::tuple const&,
            bp::tuple const&,
            bp::tuple const&,
            fern::ValueType>(
                bp::args(
                    "sizes",
                    "origin",
                    "cell_sizes",
                    "value_type"),
                "Construct a MaskedRaster"))
        .def(bp::init<
            bp::list&,
            bp::list&,
            bp::tuple&,
            bp::tuple&,
            fern::ValueType>(
                bp::args(
                    "values",
                    "mask",
                    "origin",
                    "cell_sizes",
                    "value_type"),
                "Construct a MaskedRaster"))
        .add_property("sizes", &fp::MaskedRaster::sizes,
            "Number of rows and columns")
        .add_property("origin", &fp::MaskedRaster::origin,
            "X- and y-coordinates of origin (north-west)")
        .add_property("cell_sizes", &fp::MaskedRaster::cell_sizes,
            "Cell width and cell height")
        .add_property("value_type", &fp::MaskedRaster::value_type,
            "Value type")
        // .def(bp::self += bp::self)
        // .def(bp::self + bp::self)
        ;

    bp::def("raster_as_numpy_array", fp::raster_as_numpy_array,
        "Return a Numpy array with the raster values.");
    bp::def("mask_as_numpy_array", fp::mask_as_numpy_array,
        "Return a Numpy array with the mask.");
}
