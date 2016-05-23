#pragma once
#include <iostream>
#include <boost/fusion/include/adapt_struct.hpp>
#include <boost/spirit/home/x3.hpp>

namespace fern {
namespace wkt {
namespace ast {

struct BBox
{
    double lower_left_latitude;
    double lower_left_longitude;
    double upper_right_latitude;
    double upper_right_longitude;
};


struct LengthUnit
{
    std::string name;
    double conversion_factor;
};


inline bool operator==(
    BBox const& lhs,
    BBox const& rhs)
{
    return
        lhs.lower_left_latitude == rhs.lower_left_latitude &&
        lhs.lower_left_longitude == rhs.lower_left_longitude &&
        lhs.upper_right_latitude == rhs.upper_right_latitude &&
        lhs.upper_right_longitude == rhs.upper_right_longitude
        ;
}


inline bool operator==(
    LengthUnit const& lhs,
    LengthUnit const& rhs)
{
    return
        lhs.name == rhs.name &&
        lhs.conversion_factor == rhs.conversion_factor
        ;
}



inline std::ostream& operator<<(
    std::ostream& stream,
    BBox const& bbox)
{
    return stream
        << "(" << bbox.lower_left_longitude << ", "
        << bbox.lower_left_latitude << "), "
        << "(" << bbox.upper_right_longitude << ", "
        << bbox.upper_right_latitude << ")";
}


inline std::ostream& operator<<(
    std::ostream& stream,
    LengthUnit const& length_unit)
{
    return stream
        << length_unit.name << ", "
        << length_unit.conversion_factor;
}


// template<
//     typename T>
// std::ostream& operator<<(
//     std::ostream& stream,
//     std::vector<T> const& v)
// {
//     stream << "{";
// 
//     for(auto& el : v) {
//         stream << el << " ";
//     }
// 
//     return stream << "}";
// }
// 
// 
// std::ostream& operator<<(
//     std::ostream& stream,
//     num_pc const& p)
// {
//     if(p.k) {
//         stream << p.k;
//     }
// 
//     return stream << p.sq;
// }
// 
// 
// std::ostream& operator<<(
//     std::ostream& stream,
//     num_rng const& r)
// {
//     return stream << r.pc << "-" << r.last;
// }
// 
// 
// std::ostream& operator<<(
//     std::ostream& stream,
//     ccn const& o)
// {
//     return stream << o.c << " " << o.seq;
// }

}  // namespace ast
}  // namespace wkt
}  // namespace fern


BOOST_FUSION_ADAPT_STRUCT(
    fern::wkt::ast::BBox,
    lower_left_latitude,
    lower_left_longitude,
    upper_right_latitude,
    upper_right_longitude
)

BOOST_FUSION_ADAPT_STRUCT(
    fern::wkt::ast::LengthUnit,
    name,
    conversion_factor
)
