#pragma once
#include <map>
#include "geoneric/core/string.h"


namespace geoneric {

template<
    class Value>
class Scope
{

public:

                   Scope               ()=default;

                   ~Scope              ()=default;

                   Scope               (Scope&&)=default;

    Scope&         operator=           (Scope&&)=default;

                   Scope               (Scope const&)=default;

    Scope&         operator=           (Scope const&)=default;

    size_t         size                () const;

    void           set_value           (String const& name,
                                        Value const& value);

    bool           has_value           (String const& name) const;

    Value          value               (String const& name) const;

private:

    //! Values by name.
    std::map<String, Value> _values;

};


template<
    class Value>
inline size_t Scope<Value>::size() const
{
    return _values.size();
}


template<
    class Value>
inline void Scope<Value>::set_value(
    String const& name,
    Value const& value)
{
    _values[name] = value;
}


template<
    class Value>
inline bool Scope<Value>::has_value(
    String const& name) const
{
    return _values.find(name) != _values.end();
}


template<
    class Value>
inline Value Scope<Value>::value(
    String const& name) const
{
    assert(has_value(name));
    return _values.find(name)->second;
}

} // namespace geoneric
