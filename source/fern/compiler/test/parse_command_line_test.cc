// -----------------------------------------------------------------------------
// Fern © Geoneric
//
// This file is part of Geoneric Fern which is available under the terms of
// the GNU General Public License (GPL), version 2. If you do not want to
// be bound by the terms of the GPL, you may purchase a proprietary license
// from Geoneric (http://www.geoneric.eu/contact).
// -----------------------------------------------------------------------------
#define BOOST_TEST_MODULE fern compiler
#include <boost/test/unit_test.hpp>
#include "fern/io/core/dataset.h"
#include "fern/io/io_client.h"
#include "fern/interpreter/data_sources.h"
#include "fern/interpreter/data_syncs.h"
#include "fern/compiler/parse_command_line.h"


class Support:
    public fern::IOClient
{

public:

    Support()
        : fern::IOClient()
    {
    }

};


BOOST_FIXTURE_TEST_SUITE(parse_command_line, Support)

BOOST_AUTO_TEST_CASE(parse_command_line)
{
    using namespace fern;

    std::vector<std::shared_ptr<DataSource>> data_sources;
    std::vector<std::shared_ptr<DataSync>> data_syncs;

    // No command line arguments.
    {
        int argc = 1;
        char const* argv[] = {
            "my_command"
        };
        std::vector<DataDescription> arguments;
        std::vector<DataDescription> results;

        std::tie(data_sources, data_syncs) = fern::parse_command_line(argc,
            const_cast<char**>(argv), arguments, results);
        BOOST_CHECK_EQUAL(data_sources.size(), 0u);
        BOOST_CHECK_EQUAL(data_syncs.size(), 0u);
    }

    // No command line arguments, one argument.
    {
        int argc = 1;
        char const* argv[] = {
            "my_command"
        };
        std::vector<DataDescription> arguments({
            DataDescription("my_argument")
        });
        std::vector<DataDescription> results;

        std::tie(data_sources, data_syncs) = fern::parse_command_line(argc,
            const_cast<char**>(argv), arguments, results);
        BOOST_CHECK_EQUAL(data_sources.size(), 0u);
        BOOST_CHECK_EQUAL(data_syncs.size(), 0u);
    }

    // One command line arguments, one argument.
    {
        int argc = 2;
        char const* argv[] = {
            "my_command",
            "5"
        };
        std::vector<DataDescription> arguments({
            DataDescription("my_argument")
        });
        std::vector<DataDescription> results;

        std::tie(data_sources, data_syncs) = fern::parse_command_line(argc,
            const_cast<char**>(argv), arguments, results);
        BOOST_CHECK_EQUAL(data_sources.size(), 1u);
        BOOST_CHECK_EQUAL(data_syncs.size(), 0u);

        BOOST_REQUIRE(data_sources[0]);
        std::shared_ptr<ConstantSource<int64_t>> source(
            std::dynamic_pointer_cast<ConstantSource<int64_t>>(
                data_sources[0]));
        BOOST_REQUIRE(source);
        std::shared_ptr<AttributeArgument> attribute_argument(
            std::dynamic_pointer_cast<AttributeArgument>(source->read()));
        BOOST_REQUIRE(attribute_argument);
        std::shared_ptr<ConstantAttribute<int64_t>> attribute(
            std::dynamic_pointer_cast<ConstantAttribute<int64_t>>(
                attribute_argument->attribute()));
        BOOST_REQUIRE(attribute);
        BOOST_CHECK_EQUAL(attribute->values().value(), 5);
    }

    // One command line argument, one result.
    {
        int argc = 2;
        char const* argv[] = {
            "my_command",
            "result.frn"
        };
        std::vector<DataDescription> arguments({
        });
        std::vector<DataDescription> results({
            DataDescription("my_result")
        });

        std::tie(data_sources, data_syncs) = fern::parse_command_line(argc,
            const_cast<char**>(argv), arguments, results);
        BOOST_CHECK_EQUAL(data_sources.size(), 0u);
        BOOST_CHECK_EQUAL(data_syncs.size(), 1u);

        BOOST_REQUIRE(data_syncs[0]);

        std::shared_ptr<DatasetSync> sync(
            std::dynamic_pointer_cast<DatasetSync>(data_syncs[0]));
        BOOST_REQUIRE(sync);
        BOOST_CHECK_EQUAL(sync->dataset()->name(), fern::String("result.frn"));
        BOOST_CHECK(sync->dataset()->open_mode() == OpenMode::OVERWRITE);
    }

    // Too many command line arguments.
    {
        int argc = 2;
        char const* argv[] = {
            "my_command",
            "result.frn"
        };
        std::vector<DataDescription> arguments;
        std::vector<DataDescription> results;

        BOOST_CHECK_THROW(fern::parse_command_line(argc, const_cast<char**>(
            argv), arguments, results), std::invalid_argument);
    }
}

BOOST_AUTO_TEST_SUITE_END()
