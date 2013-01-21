#pragma once
#include <memory>
#include <unordered_map>
#include "ranally/feature/geometry.h"


namespace ranally {

//! Collection for mapping feature id's to attribute values.
/*!
  This class models a collection which maps feature id's to pointers to
  attribute values. It supports creating subsets of domain item related
  values, where the subsets share the values with the superset.

  \sa        .
*/
template<
    class T>
class FidMap:
    private std::unordered_map<Fid, std::shared_ptr<T>>
{

private:

    typedef std::unordered_map<Fid, std::shared_ptr<T>> Base;

public:

    typedef typename Base::const_iterator const_iterator;

    typedef typename Base::value_type value_type;

                   FidMap              ();

                   FidMap              (FidMap const&)=delete;

    FidMap&        operator=           (FidMap const&)=delete;

                   FidMap              (FidMap&&)=delete;

    FidMap&        operator=           (FidMap&&)=delete;

    virtual        ~FidMap             ();

    const_iterator begin               () const;

    const_iterator end                 () const;

    size_t         size                () const;

    void           insert              (Fid fid,
                                        T const& value);

    T const&       at                  (Fid index) const;

private:

};


template<
    class T>
inline FidMap<T>::FidMap()

    : Base()

{
}


template<
    class T>
inline FidMap<T>::~FidMap()
{
}


template<
    class T>
inline typename FidMap<T>::const_iterator
    FidMap<T>::begin() const
{
    return Base::begin();
}


template<
    class T>
inline typename FidMap<T>::const_iterator
    FidMap<T>::end() const
{
    return Base::end();
}


template<
    class T>
inline size_t FidMap<T>::size() const
{
    return Base::size();
}


template<
    class T>
inline T const& FidMap<T>::at(
    Fid index) const
{
    return *Base::at(index);
}


template<
    class T>
inline void FidMap<T>::insert(
    Fid fid,
    T const& value)
{
    Base::operator[](fid) =
        std::make_shared<T>(value);
}

} // namespace ranally
