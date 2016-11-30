#pragma once
#include <iostream>
#include <boost/fusion/include/adapt_struct.hpp>
#include <boost/spirit/home/x3.hpp>
#include <boost/optional.hpp>
#include <boost/optional/optional_io.hpp>
#include <boost/variant.hpp>


namespace fern {
namespace wkt {
namespace ast {

struct Second
{
    uint32_t integer;
    boost::optional<uint32_t> fraction;
};


struct LocalTimeZoneDesignator
{
    char sign;
    uint32_t hour;
    boost::optional<uint32_t> minute;
};


using TimeZoneDesignator = boost::variant<char, LocalTimeZoneDesignator>;


struct GregorianCalendarDate
{
    uint32_t year;
    boost::optional<uint32_t> month;
    boost::optional<uint32_t> day;
};


struct HourClock
{
    uint32_t hour;
    boost::optional<uint32_t> minute;
    boost::optional<Second> second;
    TimeZoneDesignator time_zone;
};


struct GregorianCalendarDateTime
{
    GregorianCalendarDate date;
    boost::optional<HourClock> time;
};


struct GregorianOrdinalDate
{
    uint32_t year;
    boost::optional<uint32_t> day;
};


struct GregorianOrdinalDateTime
{
    GregorianOrdinalDate date;
    boost::optional<HourClock> time;
};


using DateTime = boost::variant<GregorianCalendarDateTime,
    GregorianOrdinalDateTime>;


struct TemporalExtent
{
    boost::variant<DateTime, std::string> start;
    boost::variant<DateTime, std::string> end;
};


struct BBox
{
    double lower_left_latitude;
    double lower_left_longitude;
    double upper_right_latitude;
    double upper_right_longitude;
};


using AuthorityUniqueIdentifier = boost::variant<double, std::string>;


using Version = boost::variant<double, std::string>;


struct Identifier
{
    std::string authority_name;
    AuthorityUniqueIdentifier authority_unique_identifier;
    Version version;
    boost::optional<std::string> authority_citation;
    boost::optional<std::string> id_uri;
};


struct AngleUnit
{
    std::string name;
    double conversion_factor;
    std::vector<Identifier> identifiers;
};


struct LengthUnit
{
    std::string name;
    double conversion_factor;
    std::vector<Identifier> identifiers;
};


struct ScaleUnit
{
    std::string name;
    double conversion_factor;
    std::vector<Identifier> identifiers;
};


struct ParametricUnit
{
    std::string name;
    double conversion_factor;
    std::vector<Identifier> identifiers;
};


struct TimeUnit
{
    std::string name;
    double conversion_factor;
    std::vector<Identifier> identifiers;
};


struct VerticalExtent
{
    double minimum_height;
    double maximum_height;
    boost::optional<LengthUnit> length_unit;
};


inline bool operator==(
    Second const& lhs,
    Second const& rhs)
{
    return
        lhs.integer == rhs.integer &&
        lhs.fraction == rhs.fraction
        ;
}


inline bool operator==(
    LocalTimeZoneDesignator const& lhs,
    LocalTimeZoneDesignator const& rhs)
{
    return
        lhs.sign == rhs.sign &&
        lhs.hour == rhs.hour &&
        lhs.minute == rhs.minute
        ;
}


inline bool operator==(
    GregorianCalendarDate const& lhs,
    GregorianCalendarDate const& rhs)
{
    return
        lhs.year == rhs.year &&
        lhs.month == rhs.month &&
        lhs.day == rhs.day
        ;
}


inline bool operator==(
    HourClock const& lhs,
    HourClock const& rhs)
{
    return
        lhs.hour == rhs.hour &&
        lhs.minute == rhs.minute &&
        lhs.second == rhs.second &&
        lhs.time_zone == rhs.time_zone
        ;
}


inline bool operator==(
    GregorianCalendarDateTime const& lhs,
    GregorianCalendarDateTime const& rhs)
{
    return
        lhs.date == rhs.date &&
        lhs.time == rhs.time
        ;
}


inline bool operator==(
    GregorianOrdinalDate const& lhs,
    GregorianOrdinalDate const& rhs)
{
    return
        lhs.year == rhs.year &&
        lhs.day == rhs.day
        ;
}


inline bool operator==(
    GregorianOrdinalDateTime const& lhs,
    GregorianOrdinalDateTime const& rhs)
{
    return
        lhs.date == rhs.date &&
        lhs.time == rhs.time
        ;
}


inline bool operator==(
    TemporalExtent const& lhs,
    TemporalExtent const& rhs)
{
    return
        lhs.start == rhs.start &&
        lhs.end == rhs.end
        ;
}


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
    AngleUnit const& lhs,
    AngleUnit const& rhs)
{
    return
        lhs.name == rhs.name &&
        lhs.conversion_factor == rhs.conversion_factor &&
        lhs.identifiers == rhs.identifiers
        ;
}


inline bool operator==(
    LengthUnit const& lhs,
    LengthUnit const& rhs)
{
    return
        lhs.name == rhs.name &&
        lhs.conversion_factor == rhs.conversion_factor &&
        lhs.identifiers == rhs.identifiers
        ;
}


inline bool operator==(
    ScaleUnit const& lhs,
    ScaleUnit const& rhs)
{
    return
        lhs.name == rhs.name &&
        lhs.conversion_factor == rhs.conversion_factor &&
        lhs.identifiers == rhs.identifiers
        ;
}


inline bool operator==(
    ParametricUnit const& lhs,
    ParametricUnit const& rhs)
{
    return
        lhs.name == rhs.name &&
        lhs.conversion_factor == rhs.conversion_factor &&
        lhs.identifiers == rhs.identifiers
        ;
}


inline bool operator==(
    TimeUnit const& lhs,
    TimeUnit const& rhs)
{
    return
        lhs.name == rhs.name &&
        lhs.conversion_factor == rhs.conversion_factor &&
        lhs.identifiers == rhs.identifiers
        ;
}


inline bool operator==(
    VerticalExtent const& lhs,
    VerticalExtent const& rhs)
{
    return
        lhs.minimum_height == rhs.minimum_height &&
        lhs.maximum_height == rhs.maximum_height &&
        lhs.length_unit == rhs.length_unit
        ;
}


inline bool operator==(
    Identifier const& lhs,
    Identifier const& rhs)
{
    return
        lhs.authority_name == rhs.authority_name &&
        lhs.authority_unique_identifier == rhs.authority_unique_identifier &&
        lhs.version == rhs.version &&
        lhs.authority_citation == rhs.authority_citation &&
        lhs.id_uri == rhs.id_uri
        ;
}


inline std::ostream& operator<<(
    std::ostream& stream,
    Second const& second)
{
    return stream
        << second.integer << "." << second.fraction;
}


inline std::ostream& operator<<(
    std::ostream& stream,
    LocalTimeZoneDesignator const& designator)
{
    return stream
        << designator.sign << designator.hour << "." << designator.minute;
}


inline std::ostream& operator<<(
    std::ostream& stream,
    GregorianCalendarDate const& date)
{
    return stream
        << date.year << '-' << date.month << '-' << date.day;
}


inline std::ostream& operator<<(
    std::ostream& stream,
    HourClock const& clock)
{
    return stream
        << clock.hour << ':' << clock.minute << ':' << clock.second
        << clock.time_zone;
}


inline std::ostream& operator<<(
    std::ostream& stream,
    GregorianCalendarDateTime const& date_time)
{
    return stream
        << date_time.date << 'T' << date_time.time;
}


inline std::ostream& operator<<(
    std::ostream& stream,
    GregorianOrdinalDate const& date)
{
    return stream
        << date.year << '-' << date.day;
}


inline std::ostream& operator<<(
    std::ostream& stream,
    GregorianOrdinalDateTime const& date_time)
{
    return stream
        << date_time.date << 'T' << date_time.time;
}


inline std::ostream& operator<<(
    std::ostream& stream,
    TemporalExtent const& extent)
{
    return stream
        << extent.start << ", " << extent.end;
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
    Identifier const& identifier)
{
    return stream
        << identifier.authority_name << ", "
        << identifier.authority_unique_identifier << ", "
        // TODO Quote version if version is a string
        << identifier.version << ", "
        << identifier.authority_citation << ", "
        << identifier.id_uri
        ;
}


inline std::ostream& operator<<(
    std::ostream& stream,
    AngleUnit const& length_unit)
{
    stream
        << length_unit.name << ", "
        << length_unit.conversion_factor;

    for(auto const& identifier: length_unit.identifiers) {
        stream << ", " << identifier;
    }

    return stream;
}


inline std::ostream& operator<<(
    std::ostream& stream,
    LengthUnit const& length_unit)
{
    stream
        << length_unit.name << ", "
        << length_unit.conversion_factor;

    for(auto const& identifier: length_unit.identifiers) {
        stream << ", " << identifier;
    }

    return stream;
}


inline std::ostream& operator<<(
    std::ostream& stream,
    ScaleUnit const& scale_unit)
{
    stream
        << scale_unit.name << ", "
        << scale_unit.conversion_factor;

    for(auto const& identifier: scale_unit.identifiers) {
        stream << ", " << identifier;
    }

    return stream;
}


inline std::ostream& operator<<(
    std::ostream& stream,
    ParametricUnit const& parametric_unit)
{
    stream
        << parametric_unit.name << ", "
        << parametric_unit.conversion_factor;

    for(auto const& identifier: parametric_unit.identifiers) {
        stream << ", " << identifier;
    }

    return stream;
}


inline std::ostream& operator<<(
    std::ostream& stream,
    TimeUnit const& time_unit)
{
    stream
        << time_unit.name << ", "
        << time_unit.conversion_factor;

    for(auto const& identifier: time_unit.identifiers) {
        stream << ", " << identifier;
    }

    return stream;
}


inline std::ostream& operator<<(
    std::ostream& stream,
    VerticalExtent const& vertical_extent)
{
    return stream
        << vertical_extent.minimum_height << ", "
        << vertical_extent.maximum_height << ", "
        << vertical_extent.length_unit;
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
    fern::wkt::ast::Second,
    integer,
    fraction
)


BOOST_FUSION_ADAPT_STRUCT(
    fern::wkt::ast::LocalTimeZoneDesignator,
    sign,
    hour,
    minute
)


BOOST_FUSION_ADAPT_STRUCT(
    fern::wkt::ast::GregorianCalendarDate,
    year,
    month,
    day
)


BOOST_FUSION_ADAPT_STRUCT(
    fern::wkt::ast::HourClock,
    hour,
    minute,
    second,
    time_zone
)


BOOST_FUSION_ADAPT_STRUCT(
    fern::wkt::ast::GregorianCalendarDateTime,
    date,
    time
)


BOOST_FUSION_ADAPT_STRUCT(
    fern::wkt::ast::GregorianOrdinalDate,
    year,
    day
)


BOOST_FUSION_ADAPT_STRUCT(
    fern::wkt::ast::GregorianOrdinalDateTime,
    date,
    time
)


BOOST_FUSION_ADAPT_STRUCT(
    fern::wkt::ast::TemporalExtent,
    start,
    end
)


BOOST_FUSION_ADAPT_STRUCT(
    fern::wkt::ast::BBox,
    lower_left_latitude,
    lower_left_longitude,
    upper_right_latitude,
    upper_right_longitude
)

BOOST_FUSION_ADAPT_STRUCT(
    fern::wkt::ast::AngleUnit,
    name,
    conversion_factor,
    identifiers
)

BOOST_FUSION_ADAPT_STRUCT(
    fern::wkt::ast::LengthUnit,
    name,
    conversion_factor,
    identifiers
)

BOOST_FUSION_ADAPT_STRUCT(
    fern::wkt::ast::ScaleUnit,
    name,
    conversion_factor,
    identifiers
)

BOOST_FUSION_ADAPT_STRUCT(
    fern::wkt::ast::ParametricUnit,
    name,
    conversion_factor,
    identifiers
)

BOOST_FUSION_ADAPT_STRUCT(
    fern::wkt::ast::TimeUnit,
    name,
    conversion_factor,
    identifiers
)

BOOST_FUSION_ADAPT_STRUCT(
    fern::wkt::ast::VerticalExtent,
    minimum_height,
    maximum_height,
    length_unit
)

BOOST_FUSION_ADAPT_STRUCT(
    fern::wkt::ast::Identifier,
    authority_name,
    authority_unique_identifier,
    version,
    authority_citation,
    id_uri
)
