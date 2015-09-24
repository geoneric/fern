// -----------------------------------------------------------------------------
// Fern © Geoneric
//
// This file is part of Geoneric Fern which is available under the terms of
// the GNU General Public License (GPL), version 2. If you do not want to
// be bound by the terms of the GPL, you may purchase a proprietary license
// from Geoneric (http://www.geoneric.eu/contact).
// -----------------------------------------------------------------------------
#pragma once
#include <bitset>
#include <string>


namespace fern {


//! Collection for managing flags.
/*!

  The implementation uses the curiously recurring template pattern in order
  to be able to define some of the methods that return a reference to the
  instance. Without returning a reference to the specialization, the original
  instance would be sliced.
*/
template<
    typename Flags,
    typename Flag,
    size_t size_>
class FlagCollection:
    private std::bitset<size_>
{

    friend class FlagCollectionTest;

public:

                   FlagCollection      ()=default;

    virtual        ~FlagCollection     ()=default;

                   FlagCollection      (FlagCollection&&)=default;

    FlagCollection& operator=          (FlagCollection&&)=default;

                   FlagCollection      (FlagCollection const&)=default;

    FlagCollection& operator=          (FlagCollection const&)=default;

    virtual std::string
                   to_string           () const=0;

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

                   FlagCollection      (unsigned long long bits);

private:

};


template<
    typename Flags,
    typename Flag,
    size_t size_>
inline FlagCollection<Flags, Flag, size_>::FlagCollection(
    unsigned long long bits)

    : std::bitset<size_>(bits)

{
}


template<
    typename Flags,
    typename Flag,
    size_t size_>
inline Flags FlagCollection<Flags, Flag, size_>::operator&(
    Flags const& flags) const
{
    Flags result(dynamic_cast<Flags const&>(*this));
    result &= flags;
    return result;
}


template<
    typename Flags,
    typename Flag,
    size_t size_>
inline Flags& FlagCollection<Flags, Flag, size_>::operator&=(
    Flags const& flags)
{
    std::bitset<size_>::operator&=(flags);
    return dynamic_cast<Flags&>(*this);
}


template<
    typename Flags,
    typename Flag,
    size_t size_>
inline Flags& FlagCollection<Flags, Flag, size_>::operator|=(
    Flags const& flags)
{
    std::bitset<size_>::operator|=(flags);
    return dynamic_cast<Flags&>(*this);
}


template<
    typename Flags,
    typename Flag,
    size_t size_>
inline Flags& FlagCollection<Flags, Flag, size_>::operator^=(
    Flags const& flags)
{
    std::bitset<size_>::operator^=(flags);
    return dynamic_cast<Flags&>(*this);
}


template<
    typename Flags,
    typename Flag,
    size_t size_>
inline size_t FlagCollection<Flags, Flag, size_>::count() const
{
    return std::bitset<size_>::count();
}


template<
    typename Flags,
    typename Flag,
    size_t size_>
inline bool FlagCollection<Flags, Flag, size_>::test(
    size_t pos) const
{
    return std::bitset<size_>::test(pos);
}


template<
    typename Flags,
    typename Flag,
    size_t size_>
inline bool FlagCollection<Flags, Flag, size_>::none() const
{
    return std::bitset<size_>::none();
}


template<
    typename Flags,
    typename Flag,
    size_t size_>
inline bool FlagCollection<Flags, Flag, size_>::any() const
{
    return std::bitset<size_>::any();
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
    typename Flags,
    typename Flag,
    size_t size_>
inline bool FlagCollection<Flags, Flag, size_>::fixed() const
{
    return count() == 1u;
}


template<
    typename Flags,
    typename Flag,
    size_t size_>
inline bool FlagCollection<Flags, Flag, size_>::is_subset_of(
    FlagCollection const& flags) const
{
    return any() && ((*this & flags).count() == count());
}


template<
    typename Flags,
    typename Flag,
    size_t size_>
inline bool FlagCollection<Flags, Flag, size_>::operator==(
    FlagCollection const& flags) const
{
    return std::bitset<size_>::operator==(flags);
}


template<
    typename Flags,
    typename Flag,
    size_t size_>
inline bool FlagCollection<Flags, Flag, size_>::operator!=(
    FlagCollection const& flags) const
{
    return std::bitset<size_>::operator!=(flags);
}

} // namespace fern
