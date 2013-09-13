#pragma once

#include <map>


namespace geoneric {

//! short_description_HORRIBLE_LONG_STRING_TO_NOTICE_THAT_IT_SHOULD_BE_REPLACED
/*!
  longer_description_HORRIBLE_LONG_STRING_TO_NOTICE_THAT_IT_SHOULD_BE_REPLACED

  \sa        .
*/
template<
    class GID,
    class Value>
class Values
{

public:

                   Values              ()=default;

                   Values              (Values const&)=delete;

    Values&        operator=           (Values const&)=delete;

                   Values              (Values&&)=delete;

    Values&        operator=           (Values&&)=delete;

                   ~Values             ()=default;

    void           add                 (GID const& gid,
                                        Value const& value);

private:

    std::map<GID, Value> _values;

};


template<
    class GID,
    class Value>
void Values::add(
    GID const& gid,
    Value const& value)
{
    _values[gid] = value;
}

} // namespace geoneric
