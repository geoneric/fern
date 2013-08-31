#pragma once
#include "ranally/core/flag_collection.h"
#include "ranally/operation/core/data_type.h"


namespace ranally {

//! Collection of data types.
/*!
  This class is for keeping track of a collection of data types. There are
  predefined collections for basic and combined data types. Member functions
  allow you to add and remove certain data types.

  \warning   Don't instantiate this class as a static variable.
             The implementation depends on static variables which may or may
             not be initialized yet (the order of initialization of static
             variables across compilation units is undefined).

  \todo Ditch SCALAR? Rename SCALAR to CONSTANT?
  \todo Rename to FeatureType?
*/
class DataTypes:
    public FlagCollection<DataTypes, DataType, DataType::DT_LAST_DATA_TYPE + 1>
{

    friend class DataTypesTest;

public:

    //! Unknown data type, no flag is set.
    static DataTypes const UNKNOWN;

    //! Scalar data type.
    static DataTypes const SCALAR;

    //! Point feature.
    static DataTypes const POINT;

    //! Line feature.
    static DataTypes const LINE;

    //! Polygon feature.
    static DataTypes const POLYGON;

    //! Alias for all feature data types.
    static DataTypes const FEATURE;

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

    constexpr      DataTypes           (unsigned long long bits);

};


constexpr inline DataTypes::DataTypes(
    unsigned long long bits)

    : FlagCollection<DataTypes, DataType, DataType::DT_LAST_DATA_TYPE + 1>(bits)

{
}


DataTypes          operator|           (DataTypes const& lhs,
                                        DataTypes const& rhs);

std::ostream&      operator<<          (std::ostream& stream,
                                        DataTypes const& flags);

} // namespace ranally
