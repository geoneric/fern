#pragma once

#include <boost/multi_array.hpp>


namespace geoneric {

boost::multi_array_types::extent_gen extents;


//! short_description_HORRIBLE_LONG_STRING_TO_NOTICE_THAT_IT_SHOULD_BE_REPLACED
/*!
  longer_description_HORRIBLE_LONG_STRING_TO_NOTICE_THAT_IT_SHOULD_BE_REPLACED

  \sa        .
*/
template<
    class T,
    size_t nr_dimensions>
class Array:
    public boost::multi_array<T, nr_dimensions>
{

public:

                   Array               ()=default;

    template<class ExtentList>
                   Array               (ExtentList const& sizes);

                   Array               (Array const&)=delete;

    Array&         operator=           (Array const&)=delete;

                   Array               (Array&&)=delete;

    Array&         operator=           (Array&&)=delete;

                   ~Array              ()=default;

private:

};


template<
    class T,
    size_t nr_dimensions>
template<
    class ExtentList>
inline Array<T, nr_dimensions>::Array(
    ExtentList const& sizes)

    : boost::multi_array<T, nr_dimensions>(sizes)

{
}

} // namespace geoneric
