#pragma once
#include <bitset>
#include <set>
#include "ranally/core/string.h"


namespace ranally {

template<
    typename Flag,
    size_t size>
class FlagCollection:
    private std::bitset<size>
{

    friend class FlagCollectionTest;

public:

                   FlagCollection      ();

    virtual        ~FlagCollection     ()=default;

                   FlagCollection      (FlagCollection&&)=default;

    FlagCollection& operator=          (FlagCollection&&)=default;

                   FlagCollection      (FlagCollection const&)=default;

    FlagCollection& operator=          (FlagCollection const&)=default;

    virtual String to_string           () const;

    FlagCollection& operator|=         (FlagCollection const& flags);

    FlagCollection& operator^=         (FlagCollection const& flags);

    bool           operator==          (FlagCollection const& flags) const;

    bool           operator!=          (FlagCollection const& flags) const;

    size_t         count               () const;

    bool           test                (size_t pos) const;

    bool           fixed               () const;

protected:

                   FlagCollection      (std::set<Flag> const& flags);

private:

};


template<
    class Flag,
    size_t size>
inline FlagCollection<Flag, size>::FlagCollection()

    : std::bitset<size>()

{
}


template<
    class Flag,
    size_t size>
inline FlagCollection<Flag, size>::FlagCollection(
    std::set<Flag> const& flags)

    : std::bitset<size>()

{
    for(Flag flag: flags) {
        std::bitset<size>::set(flag);
    }
}


template<
    class Flag,
    size_t size>
inline FlagCollection<Flag, size>& FlagCollection<Flag, size>::operator|=(
    FlagCollection const& flags)
{
    std::bitset<size>::operator|=(flags);
    return *this;
}


template<
    class Flag,
    size_t size>
inline FlagCollection<Flag, size>& FlagCollection<Flag, size>::operator^=(
    FlagCollection const& flags)
{
    std::bitset<size>::operator^=(flags);
    return *this;
}


template<
    class Flag,
    size_t size>
inline size_t FlagCollection<Flag, size>::count() const
{
    return std::bitset<size>::count();
}


template<
    class Flag,
    size_t size>
inline bool FlagCollection<Flag, size>::test(
    size_t pos) const
{
    return std::bitset<size>::test(pos);
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
    class Flag,
    size_t size>
inline bool FlagCollection<Flag, size>::fixed() const
{
    return count() == 1u;
}


template<
    class Flag,
    size_t size>
inline bool FlagCollection<Flag, size>::operator==(
    FlagCollection const& flags) const
{
    return std::bitset<size>::operator==(
        flags);
}


template<
    class Flag,
    size_t size>
inline bool FlagCollection<Flag, size>::operator!=(
    FlagCollection const& flags) const
{
    return std::bitset<size>::operator!=(
        flags);
}


template<
    class Flag,
    size_t size>
inline String FlagCollection<Flag, size>::to_string() const
{
    return std::bitset<size>::to_string();
}


template<
    class Flag,
    size_t size>
inline FlagCollection<Flag, size> operator|(
    FlagCollection<Flag, size> const& lhs,
    FlagCollection<Flag, size> const& rhs)
{
    FlagCollection<Flag, size> result(lhs);
    result |= rhs;
    return result;
}


template<
    class Flag,
    size_t size>
inline std::ostream& operator<<(
    std::ostream& stream,
    FlagCollection<Flag, size> const& flags)
{
    stream << flags.to_string();
    return stream;
}

} // namespace ranally
