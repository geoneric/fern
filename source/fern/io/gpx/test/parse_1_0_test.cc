// -----------------------------------------------------------------------------
// Fern © Geoneric
//
// This file is part of Geoneric Fern which is available under the terms of
// the GNU General Public License (GPL), version 2. If you do not want to
// be bound by the terms of the GPL, you may purchase a proprietary license
// from Geoneric (http://www.geoneric.eu/contact).
// -----------------------------------------------------------------------------
#define BOOST_TEST_MODULE fern io gpx parse_1_0
#include <iostream>
#include <boost/test/unit_test.hpp>
#include "fern/core/io_error.h"
#include "fern/core/parse_error.h"
#include "fern/io/core/file.h"
#include "fern/io/gpx/parse.h"


namespace fi = fern::io;
namespace fig = fern::io::gpx_1_0;


BOOST_AUTO_TEST_SUITE(parse)

BOOST_AUTO_TEST_CASE(does_not_exist)
{
    std::string pathname = "does_not_exist.gpx";
    BOOST_REQUIRE(!fi::file_exists(pathname));
    BOOST_CHECK_THROW(fig::parse(pathname), fern::IOError);
}


BOOST_AUTO_TEST_CASE(unreadable)
{
    std::string pathname = "unreadable.gpx";
    BOOST_REQUIRE(fi::file_exists(pathname));
    BOOST_CHECK_THROW(fig::parse(pathname), fern::IOError);
}


BOOST_AUTO_TEST_CASE(invalid)
{
    std::string pathname = "invalid.gpx";
    BOOST_REQUIRE(fi::file_exists(pathname));

    try {
        fig::parse(pathname);
        BOOST_CHECK(false);
    }
    catch(fern::ParseError const& exception) {
        BOOST_CHECK_EQUAL(exception.message(),
            "Error parsing invalid.gpx:14:16: "
            "no declaration found for element 'blah'");
    }
}


BOOST_AUTO_TEST_CASE(valid)
{
    std::string pathname = "example-1.gpx";
    BOOST_REQUIRE(fi::file_exists(pathname));

    auto gpx = fig::parse(pathname);

    BOOST_REQUIRE(gpx);
    BOOST_CHECK_EQUAL(gpx->name().get(), "Example gpx");
    BOOST_CHECK_EQUAL(gpx->trk().size(), 1);

    auto const& track(gpx->trk()[0]);
    BOOST_CHECK_EQUAL(track.name().get(), "Example trk");

    BOOST_REQUIRE_EQUAL(track.trkseg().size(), 1);
    auto const& sequence(track.trkseg()[0].trkpt());
    BOOST_CHECK_EQUAL(sequence.size(), 7);

    BOOST_REQUIRE(sequence[2].ele().present());
    BOOST_CHECK_EQUAL(sequence[2].ele().get(), 2372);
    BOOST_CHECK_EQUAL(sequence[3].lat(), 46.57650000);
    BOOST_REQUIRE(sequence[6].time().present());

    auto const& date_time(sequence[6].time().get());
    BOOST_CHECK_EQUAL(date_time.year(), 2007);
    BOOST_CHECK_EQUAL(date_time.month(), 10);
    BOOST_CHECK_EQUAL(date_time.day(), 14);
    BOOST_CHECK_EQUAL(date_time.hours(), 10);
    BOOST_CHECK_EQUAL(date_time.minutes(), 14);
    BOOST_CHECK_EQUAL(date_time.seconds(), 8);

    BOOST_CHECK(date_time.zone_present());
    BOOST_CHECK_EQUAL(date_time.zone_hours(), 0);
    BOOST_CHECK_EQUAL(date_time.zone_minutes(), 0);
}

BOOST_AUTO_TEST_SUITE_END()
