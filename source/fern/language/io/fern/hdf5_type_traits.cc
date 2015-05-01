// -----------------------------------------------------------------------------
// Fern © Geoneric
//
// This file is part of Geoneric Fern which is available under the terms of
// the GNU General Public License (GPL), version 2. If you do not want to
// be bound by the terms of the GPL, you may purchase a proprietary license
// from Geoneric (http://www.geoneric.eu/contact).
// -----------------------------------------------------------------------------
#include "fern/language/io/fern/hdf5_type_traits.h"


namespace fern {
namespace language {

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

} // namespace language
} // namespace fern
