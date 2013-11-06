#pragma once
#include <map>
#include "geoneric/core/string.h"


namespace geoneric {

template<
    class Value>
class Scope
{

public:

    typedef typename std::map<String, Value>::const_iterator const_iterator;

                   Scope               ()=default;

                   ~Scope              ()=default;

                   Scope               (Scope&& other);

    Scope&         operator=           (Scope&& other);

                   Scope               (Scope const& other);

    Scope&         operator=           (Scope const& other);

    const_iterator begin               () const;

    const_iterator end                 () const;

    size_t         size                () const;

    void           set_value           (String const& name,
                                        Value const& value);

    void           erase_value         (String const& name);

    bool           has_value           (String const& name) const;

    Value          value               (String const& name) const;

    void           clear               ();

private:

    //! Values by name.
    std::map<String, Value> _values;

};


template<
    class Value>
Scope<Value>::Scope(
    Scope&& other)

    : _values()

{
    *this = std::move(other);
}


template<
    class Value>
Scope<Value>& Scope<Value>::operator=(
    Scope&& other)
{
    if(this != &other) {
        _values = std::move(other._values);
    }

    return *this;
}


template<
    class Value>
Scope<Value>::Scope(
    Scope const& other)

    : _values(other._values)

{
}


template<
    class Value>
Scope<Value>& Scope<Value>::operator=(
    Scope const& other)
{
    if(this != &other) {
        _values = other._values;
    }

    return *this;
}


template<
    class Value>
inline typename Scope<Value>::const_iterator Scope<Value>::begin() const
{
    return _values.begin();
}


template<
    class Value>
inline typename Scope<Value>::const_iterator Scope<Value>::end() const
{
    return _values.end();
}


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
inline void Scope<Value>::erase_value(
    String const& name)
{
    assert(has_value(name));
    _values.erase(name);
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


template<
    class Value>
inline void Scope<Value>::clear()
{
    return _values.clear();
}

} // namespace geoneric
