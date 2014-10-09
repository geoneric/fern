#pragma once
#include <map>
#include "fern/feature/core/values.h"


namespace fern {

//! short_description_HORRIBLE_LONG_STRING_TO_NOTICE_THAT_IT_SHOULD_BE_REPLACED
/*!
  longer_description_HORRIBLE_LONG_STRING_TO_NOTICE_THAT_IT_SHOULD_BE_REPLACED

  \sa        .
*/
template<
    typename GID,
    typename Value>
class GeometryValues:
    public Values
{

public:

    using const_iterator = typename std::map<GID, Value>::const_iterator;

                   GeometryValues      ()=default;

                   GeometryValues      (GeometryValues const&)=delete;

    GeometryValues& operator=          (GeometryValues const&)=delete;

                   GeometryValues      (GeometryValues&&)=delete;

    GeometryValues& operator=          (GeometryValues&&)=delete;

                   ~GeometryValues     ()=default;

    void           add                 (GID const& gid,
                                        Value const& value);

    bool           empty               () const;

    size_t         size                () const;

    Value const&   value               (GID const& gid) const;

    const_iterator cbegin              () const;

private:

    std::map<GID, Value> _values;

};


template<
    typename GID,
    typename Value>
inline void GeometryValues<GID, Value>::add(
    GID const& gid,
    Value const& value)
{
    _values[gid] = value;
}


template<
    typename GID,
    typename Value>
inline bool GeometryValues<GID, Value>::empty() const
{
    return _values.empty();
}


template<
    typename GID,
    typename Value>
inline size_t GeometryValues<GID, Value>::size() const
{
    return _values.size();
}


template<
    typename GID,
    typename Value>
inline Value const& GeometryValues<GID, Value>::value(
    GID const& gid) const
{
    auto iterator = _values.find(gid);
    assert(iterator != _values.end());
    return iterator->second;
}


template<
    typename GID,
    typename Value>
inline auto GeometryValues<GID, Value>::cbegin() const -> const_iterator
{
    return _values.cbegin();
}

} // namespace fern
