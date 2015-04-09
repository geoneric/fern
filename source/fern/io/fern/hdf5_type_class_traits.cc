// -----------------------------------------------------------------------------
// Fern © Geoneric
//
// This file is part of Geoneric Fern which is available under the terms of
// the GNU General Public License (GPL), version 2. If you do not want to
// be bound by the terms of the GPL, you may purchase a proprietary license
// from Geoneric (http://www.geoneric.eu/contact).
// -----------------------------------------------------------------------------
#include "fern/io/fern/hdf5_type_class_traits.h"


namespace fern {

String const HDF5TypeClassTraits<H5T_INTEGER>::name = "H5T_INTEGER";
String const HDF5TypeClassTraits<H5T_FLOAT>::name = "H5T_FLOAT";
String const HDF5TypeClassTraits<H5T_TIME>::name = "H5T_TIME";
String const HDF5TypeClassTraits<H5T_STRING>::name = "H5T_STRING";
String const HDF5TypeClassTraits<H5T_NO_CLASS>::name = "H5T_NO_CLASS";
String const HDF5TypeClassTraits<H5T_BITFIELD>::name = "H5T_BITFIELD";
String const HDF5TypeClassTraits<H5T_OPAQUE>::name = "H5T_OPAQUE";
String const HDF5TypeClassTraits<H5T_COMPOUND>::name = "H5T_COMPOUND";
String const HDF5TypeClassTraits<H5T_REFERENCE>::name = "H5T_REFERENCE";
String const HDF5TypeClassTraits<H5T_ENUM>::name = "H5T_ENUM";
String const HDF5TypeClassTraits<H5T_VLEN>::name = "H5T_VLEN";
String const HDF5TypeClassTraits<H5T_ARRAY>::name = "H5T_ARRAY";
String const HDF5TypeClassTraits<H5T_NCLASSES>::name = "H5T_NCLASSES";

} // namespace fern
