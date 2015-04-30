// -----------------------------------------------------------------------------
// Fern © Geoneric
//
// This file is part of Geoneric Fern which is available under the terms of
// the GNU General Public License (GPL), version 2. If you do not want to
// be bound by the terms of the GPL, you may purchase a proprietary license
// from Geoneric (http://www.geoneric.eu/contact).
// -----------------------------------------------------------------------------
#include "fern/language/io/fern/hdf5_type_class_traits.h"


namespace fern {

std::string const HDF5TypeClassTraits<H5T_INTEGER>::name = "H5T_INTEGER";
std::string const HDF5TypeClassTraits<H5T_FLOAT>::name = "H5T_FLOAT";
std::string const HDF5TypeClassTraits<H5T_TIME>::name = "H5T_TIME";
std::string const HDF5TypeClassTraits<H5T_STRING>::name = "H5T_STRING";
std::string const HDF5TypeClassTraits<H5T_NO_CLASS>::name = "H5T_NO_CLASS";
std::string const HDF5TypeClassTraits<H5T_BITFIELD>::name = "H5T_BITFIELD";
std::string const HDF5TypeClassTraits<H5T_OPAQUE>::name = "H5T_OPAQUE";
std::string const HDF5TypeClassTraits<H5T_COMPOUND>::name = "H5T_COMPOUND";
std::string const HDF5TypeClassTraits<H5T_REFERENCE>::name = "H5T_REFERENCE";
std::string const HDF5TypeClassTraits<H5T_ENUM>::name = "H5T_ENUM";
std::string const HDF5TypeClassTraits<H5T_VLEN>::name = "H5T_VLEN";
std::string const HDF5TypeClassTraits<H5T_ARRAY>::name = "H5T_ARRAY";
std::string const HDF5TypeClassTraits<H5T_NCLASSES>::name = "H5T_NCLASSES";

} // namespace fern
