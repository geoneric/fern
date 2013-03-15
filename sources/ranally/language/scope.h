#pragma once
#include <map>
#include "ranally/core/string.h"


namespace ranally {

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

    void           add_value           (String const& name,
                                        Value const& value);

    bool           has_value           (String const& name) const;

    Value const&   value               (String const& name) const;

private:

    //! Values by name.
    std::map<String, Value> _values;

};


template<
    class Value>
inline void Scope<Value>::add_value(
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
inline Value const& Scope<Value>::value(
    String const& name) const
{
    assert(has_value(name));
    return _values[name];
}

} // namespace ranally
