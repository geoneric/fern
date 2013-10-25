#include "geoneric/io/geoneric/hdf5_type_traits.h"


namespace geoneric {

H5::PredType const HDF5TypeTraits<uint8_t>::data_type =
    H5::PredType::NATIVE_UINT8;
H5::PredType const HDF5TypeTraits<uint16_t>::data_type =
    H5::PredType::NATIVE_UINT16;
H5::PredType const HDF5TypeTraits<uint32_t>::data_type =
    H5::PredType::NATIVE_UINT32;
H5::PredType const HDF5TypeTraits<uint64_t>::data_type =
    H5::PredType::NATIVE_UINT64;
H5::PredType const HDF5TypeTraits<int8_t>::data_type =
    H5::PredType::NATIVE_INT8;
H5::PredType const HDF5TypeTraits<int16_t>::data_type =
    H5::PredType::NATIVE_INT16;
H5::PredType const HDF5TypeTraits<int32_t>::data_type =
    H5::PredType::NATIVE_INT32;
H5::PredType const HDF5TypeTraits<int64_t>::data_type =
    H5::PredType::NATIVE_INT64;
H5::PredType const HDF5TypeTraits<float>::data_type =
    H5::PredType::NATIVE_FLOAT;
H5::PredType const HDF5TypeTraits<double>::data_type =
    H5::PredType::NATIVE_DOUBLE;

} // namespace geoneric
