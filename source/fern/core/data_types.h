// -----------------------------------------------------------------------------
// Fern © Geoneric
//
// This file is part of Geoneric Fern which is available under the terms of
// the GNU General Public License (GPL), version 2. If you do not want to
// be bound by the terms of the GPL, you may purchase a proprietary license
// from Geoneric (http://www.geoneric.eu/contact).
// -----------------------------------------------------------------------------
#pragma once
#include "fern/core/data_type.h"
#include "fern/core/flag_collection.h"


namespace fern {

//! Collection of data types.
/*!
  This class is for keeping track of a collection of data types. There are
  predefined collections for basic and combined data types. Member functions
  allow you to add and remove certain data types.

  \warning   Don't instantiate this class as a static variable.
             The implementation depends on static variables which may or may
             not be initialized yet (the order of initialization of static
             variables across compilation units is undefined).

  \todo Rename to FeatureType?
*/
class DataTypes:
    public FlagCollection<DataTypes, DataType, DataType::DT_LAST_DATA_TYPE + 1>
{

    friend class DataTypesTest;

public:

    //! Unknown data type, no flag is set.
    static DataTypes const UNKNOWN;

    //! Constant data type.
    static DataTypes const CONSTANT;

    static DataTypes const STATIC_FIELD;

    // //! Point feature.
    // static DataTypes const POINT;

    // //! Line feature.
    // static DataTypes const LINE;

    // //! Polygon feature.
    // static DataTypes const POLYGON;

    // //! Alias for all feature data types.
    // static DataTypes const FEATURE;

    //! Alias for all data types.
    static DataTypes const ALL;

    static DataTypes from_string       (String const& string);

                   DataTypes           ();

                   ~DataTypes          ()=default;

                   DataTypes           (DataTypes&&)=default;

    DataTypes&     operator=           (DataTypes&&)=default;

                   DataTypes           (DataTypes const&)=default;

    DataTypes&     operator=           (DataTypes const&)=default;

    String         to_string           () const;

private:

    static std::vector<DataType> const DATA_TYPES;

                   DataTypes           (unsigned long long bits);

};


inline DataTypes::DataTypes(
    unsigned long long bits)

    : FlagCollection<DataTypes, DataType, DataType::DT_LAST_DATA_TYPE + 1>(bits)

{
}


DataTypes          operator|           (DataTypes const& lhs,
                                        DataTypes const& rhs);

std::ostream&      operator<<          (std::ostream& stream,
                                        DataTypes const& flags);

} // namespace fern
