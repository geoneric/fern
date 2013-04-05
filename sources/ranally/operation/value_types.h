#pragma once
#include "ranally/core/flag_collection.h"
#include "ranally/operation/value_type.h"


namespace ranally {

//! short_description_HORRIBLE_LONG_STRING_TO_NOTICE_THAT_IT_SHOULD_BE_REPLACED
/*!
  longer_description_HORRIBLE_LONG_STRING_TO_NOTICE_THAT_IT_SHOULD_BE_REPLACED

  \todo      Refactor with DataTypes.
  \sa        .
*/
class ValueTypes:
    public FlagCollection<ValueTypes, ValueType,
        ValueType::VT_LAST_VALUE_TYPE + 1>
{

    friend class ValueTypesTest;

public:

    //! Unknown value type, no flag is set.
    static ValueTypes const UNKNOWN;

    static ValueTypes const UINT8;

    static ValueTypes const INT8;

    static ValueTypes const UINT16;

    static ValueTypes const INT16;

    static ValueTypes const UINT32;

    static ValueTypes const INT32;

    static ValueTypes const UINT64;

    static ValueTypes const INT64;

    static ValueTypes const FLOAT32;

    static ValueTypes const FLOAT64;

    static ValueTypes const STRING;

    //! Alias for all unsigned integer value types.
    static ValueTypes const UNSIGNED_INTEGER;

    //! Alias for all signed integer value types.
    static ValueTypes const SIGNED_INTEGER;

    //! Alias for all integer value types.
    static ValueTypes const INTEGER;

    //! Alias for a large unsigned integer type.
    static ValueTypes const SIZE;

    //! Alias for all floating point value types.
    static ValueTypes const FLOATING_POINT;

    //! Alias for all numeric value types.
    static ValueTypes const NUMBER;

    //! Alias for all value types.
    static ValueTypes const ALL;

    static ValueTypes from_string      (String const& string);

                   ValueTypes          ();

                   ~ValueTypes         ()=default;

                   ValueTypes          (ValueTypes&&)=default;

    ValueTypes&    operator=           (ValueTypes&&)=default;

                   ValueTypes          (ValueTypes const&)=default;

    ValueTypes&    operator=           (ValueTypes const&)=default;

    String         to_string           () const;

private:

    static std::vector<ValueType> const VALUE_TYPES;

    constexpr      ValueTypes          (unsigned long long bits);

};


constexpr inline ValueTypes::ValueTypes(
    unsigned long long bits)

    : FlagCollection<ValueTypes, ValueType,
          ValueType::VT_LAST_VALUE_TYPE + 1>(bits)

{
}


ValueTypes         operator|           (ValueTypes const& lhs,
                                        ValueTypes const& rhs);

std::ostream&      operator<<          (std::ostream& stream,
                                        ValueTypes const& flags);

} // namespace ranally
