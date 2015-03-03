#include "fern/python_extension/algorithm/numpy/wrapped_data_type.h"
#include "fern/python_extension/algorithm/core/util.h"
#include "fern/python_extension/algorithm/numpy/util.h"


namespace fern {
namespace python {

WrappedDataType data_type(
    PyObject* object)
{
    WrappedDataType result{WrappedDataType::unsupported};

    if(is_python_int(object)) {
        result = WrappedDataType::python_int;
    }
    else if(is_python_float(object)) {
        result = WrappedDataType::python_float;
    }
    else if(is_numpy_array(object)) {
        result = WrappedDataType::numpy_array;
    }

    return result;
}

} // namespace python
} // namespace fern
