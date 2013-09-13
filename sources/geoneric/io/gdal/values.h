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

    bool           empty               () const;

    size_t         size                () const;

    Value const&   value               (GID const& gid) const;

private:

    std::map<GID, Value> _values;

};


template<
    class GID,
    class Value>
inline void Values<GID, Value>::add(
    GID const& gid,
    Value const& value)
{
    _values[gid] = value;
}


template<
    class GID,
    class Value>
inline bool Values<GID, Value>::empty() const
{
    return _values.empty();
}


template<
    class GID,
    class Value>
inline size_t Values<GID, Value>::size() const
{
    return _values.size();
}


template<
    class GID,
    class Value>
inline Value const& Values<GID, Value>::value(
    GID const& gid) const
{
    auto iterator = _values.find(gid);
    assert(iterator != _values.end());
    return iterator->second;
}

} // namespace geoneric
