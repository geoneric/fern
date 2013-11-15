#pragma once
#include <bitset>
#include "fern/core/string.h"


namespace fern {


//! Collection for managing flags.
/*!

  The implementation uses the curiously recurring template pattern in order
  to be able to define some of the methods that return a reference to the
  instance. Without returning a reference to the specialization, the original
  instance would be sliced.
*/
template<
    class Flags,
    typename Flag,
    size_t size>
class FlagCollection:
    private std::bitset<size>
{

    friend class FlagCollectionTest;

public:

                   FlagCollection      ()=default;

    virtual        ~FlagCollection     ()=default;

                   FlagCollection      (FlagCollection&&)=default;

    FlagCollection& operator=          (FlagCollection&&)=default;

                   FlagCollection      (FlagCollection const&)=default;

    FlagCollection& operator=          (FlagCollection const&)=default;

    virtual String to_string           () const=0;

    Flags          operator&           (Flags const& flags) const;

    Flags&         operator&=          (Flags const& flags);

    Flags&         operator|=          (Flags const& flags);

    Flags&         operator^=          (Flags const& flags);

    bool           operator==          (FlagCollection const& flags) const;

    bool           operator!=          (FlagCollection const& flags) const;

    size_t         count               () const;

    bool           test                (size_t pos) const;

    bool           none                () const;

    bool           any                 () const;

    bool           fixed               () const;

    bool           is_subset_of        (FlagCollection const& flags) const;

protected:

    constexpr      FlagCollection      (unsigned long long bits);

private:

};


template<
    class Flags,
    typename Flag,
    size_t size>
inline constexpr FlagCollection<Flags, Flag, size>::FlagCollection(
    unsigned long long bits)

    : std::bitset<size>(bits )

{
}


template<
    class Flags,
    typename Flag,
    size_t size>
inline Flags FlagCollection<Flags, Flag, size>::operator&(
    Flags const& flags) const
{
    Flags result(dynamic_cast<Flags const&>(*this));
    result &= flags;
    return result;
}


template<
    class Flags,
    typename Flag,
    size_t size>
inline Flags& FlagCollection<Flags, Flag, size>::operator&=(
    Flags const& flags)
{
    std::bitset<size>::operator&=(flags);
    return dynamic_cast<Flags&>(*this);
}


template<
    class Flags,
    typename Flag,
    size_t size>
inline Flags& FlagCollection<Flags, Flag, size>::operator|=(
    Flags const& flags)
{
    std::bitset<size>::operator|=(flags);
    return dynamic_cast<Flags&>(*this);
}


template<
    class Flags,
    typename Flag,
    size_t size>
inline Flags& FlagCollection<Flags, Flag, size>::operator^=(
    Flags const& flags)
{
    std::bitset<size>::operator^=(flags);
    return dynamic_cast<Flags&>(*this);
}


template<
    class Flags,
    typename Flag,
    size_t size>
inline size_t FlagCollection<Flags, Flag, size>::count() const
{
    return std::bitset<size>::count();
}


template<
    class Flags,
    typename Flag,
    size_t size>
inline bool FlagCollection<Flags, Flag, size>::test(
    size_t pos) const
{
    return std::bitset<size>::test(pos);
}


template<
    class Flags,
    typename Flag,
    size_t size>
inline bool FlagCollection<Flags, Flag, size>::none() const
{
    return std::bitset<size>::none();
}


template<
    class Flags,
    typename Flag,
    size_t size>
inline bool FlagCollection<Flags, Flag, size>::any() const
{
    return std::bitset<size>::any();
}


//! Return whether only a single flag is set.
/*!
  \return    true or false
  \sa        count()

  Flag collections are used to set multiple flags. Over time some flags may
  be turned off again. If only a single flag remains, the collection is set
  to be fixed, which means that this is the final setting.
*/
template<
    class Flags,
    typename Flag,
    size_t size>
inline bool FlagCollection<Flags, Flag, size>::fixed() const
{
    return count() == 1u;
}


template<
    class Flags,
    typename Flag,
    size_t size>
inline bool FlagCollection<Flags, Flag, size>::is_subset_of(
    FlagCollection const& flags) const
{
    return any() && ((*this & flags).count() == count());
}


template<
    class Flags,
    typename Flag,
    size_t size>
inline bool FlagCollection<Flags, Flag, size>::operator==(
    FlagCollection const& flags) const
{
    return std::bitset<size>::operator==(flags);
}


template<
    class Flags,
    typename Flag,
    size_t size>
inline bool FlagCollection<Flags, Flag, size>::operator!=(
    FlagCollection const& flags) const
{
    return std::bitset<size>::operator!=(flags);
}

} // namespace fern
