#pragma once

#include <map>
#include "geoneric/io/gdal/values.h"


namespace geoneric {

//! short_description_HORRIBLE_LONG_STRING_TO_NOTICE_THAT_IT_SHOULD_BE_REPLACED
/*!
  longer_description_HORRIBLE_LONG_STRING_TO_NOTICE_THAT_IT_SHOULD_BE_REPLACED

  \sa        .
*/
template<
    class GID,
    class Value>
class GeometryValues:
    public Values
{

public:

                   GeometryValues      ()=default;

                   GeometryValues      (GeometryValues const&)=delete;

    GeometryValues&        operator=   (GeometryValues const&)=delete;

                   GeometryValues      (GeometryValues&&)=delete;

    GeometryValues&        operator=   (GeometryValues&&)=delete;

                   ~GeometryValues     ()=default;

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
inline void GeometryValues<GID, Value>::add(
    GID const& gid,
    Value const& value)
{
    _values[gid] = value;
}


template<
    class GID,
    class Value>
inline bool GeometryValues<GID, Value>::empty() const
{
    return _values.empty();
}


template<
    class GID,
    class Value>
inline size_t GeometryValues<GID, Value>::size() const
{
    return _values.size();
}


template<
    class GID,
    class Value>
inline Value const& GeometryValues<GID, Value>::value(
    GID const& gid) const
{
    auto iterator = _values.find(gid);
    assert(iterator != _values.end());
    return iterator->second;
}

} // namespace geoneric
