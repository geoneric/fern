#pragma once
#include "ranally/core/flag_collection.h"
#include "ranally/operation/data_type.h"


namespace ranally {

//! Collection of data types.
/*!
  This class is for keeping track of a collection of data types. There are
  predefined collections for basic and combined data types. Member functions
  allow you to add and remove certain data types.

  \todo      Refactor with ValueTypes.
  \sa        .
*/
class DataTypes:
    public FlagCollection<detail::DataType, detail::DataType::DT_NR_DATA_TYPES>
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

    //! Data type depends on the data type of the operation's input parameters.
    static DataTypes const DEPENDS_ON_INPUT;

    static DataTypes from_string       (String const& string);

                   DataTypes           ();

                   ~DataTypes          () noexcept(true) =default;

                   DataTypes           (DataTypes&&)=default;

    DataTypes&     operator=           (DataTypes&&)=default;

                   DataTypes           (DataTypes const&)=default;

    DataTypes&     operator=           (DataTypes const&)=default;

    String         to_string           () const;

private:

    static std::vector<detail::DataType> const DATA_TYPES;

                   DataTypes           (
                             std::set<detail::DataType> const& data_types);

};

} // namespace ranally
