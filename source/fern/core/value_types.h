// -----------------------------------------------------------------------------
// Fern © Geoneric
//
// This file is part of Geoneric Fern which is available under the terms of
// the GNU General Public License (GPL), version 2. If you do not want to
// be bound by the terms of the GPL, you may purchase a proprietary license
// from Geoneric (http://www.geoneric.eu/contact).
// -----------------------------------------------------------------------------
#pragma once
#include <vector>
#include "fern/core/flag_collection.h"
#include "fern/core/value_type.h"


namespace fern {

//! short_description_HORRIBLE_LONG_STRING_TO_NOTICE_THAT_IT_SHOULD_BE_REPLACED
/*!
  longer_description_HORRIBLE_LONG_STRING_TO_NOTICE_THAT_IT_SHOULD_BE_REPLACED

  \warning   Don't instantiate this class as a static variable.
             The implementation depends on static variables which may or may
             not be initialized yet (the order of initialization of static
             variables across compilation units is undefined).
*/
class ValueTypes:
    public FlagCollection<ValueTypes, ValueType,
        ValueType::VT_LAST_VALUE_TYPE + 1>
{

    friend class ValueTypesTest;

public:

    //! Unknown value type, no flag is set.
    static ValueTypes const UNKNOWN;

    static ValueTypes const BOOL;

    static ValueTypes const CHAR;

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

    static ValueTypes from_string      (std::string const& string);

                   ValueTypes          ();

                   ~ValueTypes         ()=default;

                   ValueTypes          (ValueTypes&&)=default;

    ValueTypes&    operator=           (ValueTypes&&)=default;

                   ValueTypes          (ValueTypes const&)=default;

    ValueTypes&    operator=           (ValueTypes const&)=default;

    std::string    to_string           () const;

private:

    static std::vector<ValueType> const VALUE_TYPES;

                   ValueTypes          (unsigned long long bits);

};


inline ValueTypes::ValueTypes(
    unsigned long long bits)

    : FlagCollection<ValueTypes, ValueType,
          ValueType::VT_LAST_VALUE_TYPE + 1>(bits)

{
}


ValueTypes         operator|           (ValueTypes const& lhs,
                                        ValueTypes const& rhs);

std::ostream&      operator<<          (std::ostream& stream,
                                        ValueTypes const& flags);

} // namespace fern
