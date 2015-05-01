// -----------------------------------------------------------------------------
// Fern © Geoneric
//
// This file is part of Geoneric Fern which is available under the terms of
// the GNU General Public License (GPL), version 2. If you do not want to
// be bound by the terms of the GPL, you may purchase a proprietary license
// from Geoneric (http://www.geoneric.eu/contact).
// -----------------------------------------------------------------------------
#pragma once
#include <cassert>
#include <map>
#include <string>


namespace fern {

template<
    class Value>
class Scope
{

public:

    using const_iterator =
        typename std::map<std::string, Value>::const_iterator;

                   Scope               ()=default;

                   ~Scope              ()=default;

                   Scope               (Scope&& other);

    Scope&         operator=           (Scope&& other);

                   Scope               (Scope const& other);

    Scope&         operator=           (Scope const& other);

    const_iterator begin               () const;

    const_iterator end                 () const;

    size_t         size                () const;

    void           set_value           (std::string const& name,
                                        Value const& value);

    void           erase_value         (std::string const& name);

    bool           has_value           (std::string const& name) const;

    Value          value               (std::string const& name) const;

    void           clear               ();

private:

    //! Values by name.
    std::map<std::string, Value> _values;

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
    std::string const& name,
    Value const& value)
{
    _values[name] = value;
}


template<
    class Value>
inline void Scope<Value>::erase_value(
    std::string const& name)
{
    assert(has_value(name));
    _values.erase(name);
}


template<
    class Value>
inline bool Scope<Value>::has_value(
    std::string const& name) const
{
    return _values.find(name) != _values.end();
}


template<
    class Value>
inline Value Scope<Value>::value(
    std::string const& name) const
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

} // namespace fern
