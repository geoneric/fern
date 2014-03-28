#pragma once
// TODO Disable range checks in release builds.
// #define BOOST_DISABLE_ASSERTS
#include <boost/multi_array.hpp>


namespace fern {

extern boost::multi_array_types::extent_gen extents;

extern boost::multi_array_types::index_gen indices;

typedef boost::multi_array_types::index_range Range;


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

                   Array               (std::initializer_list<
                                           std::initializer_list<T>> const&
                                              values);

    template<class ExtentList>
                   Array               (ExtentList const& sizes);

                   Array               (Array const&)=default;

    Array&         operator=           (Array const&)=default;

                   Array               (Array&&)=default;

    Array&         operator=           (Array&&)=default;

    virtual        ~Array              ()=default;

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


template<
    class T,
    size_t nr_dimensions>
inline Array<T, nr_dimensions>::Array(
    std::initializer_list<std::initializer_list<T>> const& values)

    : boost::multi_array<T, nr_dimensions>(
          extents[values.size()][values.begin()->size()])

{
    T* it = this->data();

    for(auto const& row: values) {
        for(auto const& col: row) {
            *it++ = col;
        }
    }
}

} // namespace fern
