// -----------------------------------------------------------------------------
// Fern © Geoneric
//
// This file is part of Geoneric Fern which is available under the terms of
// the GNU General Public License (GPL), version 2. If you do not want to
// be bound by the terms of the GPL, you may purchase a proprietary license
// from Geoneric (http://www.geoneric.eu/contact).
// -----------------------------------------------------------------------------
#define BOOST_TEST_MODULE fern core
#include <boost/any.hpp>
#include <boost/test/unit_test.hpp>
#include "fern/language/core/symbol_table.h"


BOOST_AUTO_TEST_SUITE(symbol_table)


BOOST_AUTO_TEST_CASE(scoping)
{
    using namespace fern;

    using SymbolTable = SymbolTable<boost::any>;

    SymbolTable table;
    std::string name("a");
    BOOST_CHECK_EQUAL(table.scope_level(), SymbolTable::size_type(0));

    // Add and remove one value for 'a'.
    {
        table.push_scope();
        BOOST_CHECK_EQUAL(table.scope_level(), SymbolTable::size_type(1));

        table.add_value(name, boost::any(5));
        BOOST_CHECK_EQUAL(table.size(), 1u);
        BOOST_REQUIRE(table.has_value(name));
        BOOST_CHECK_EQUAL(boost::any_cast<int>(table.value(name)), 5);
        BOOST_CHECK_EQUAL(table.scope_level(name), table.scope_level());

        table.pop_scope();
        BOOST_REQUIRE(!table.has_value(name));

        BOOST_CHECK_EQUAL(table.scope_level(), SymbolTable::size_type(0));
    }

    // Add and remove two definitions for 'a', not nested.
    {
        table.push_scope();
        BOOST_CHECK_EQUAL(table.scope_level(), SymbolTable::size_type(1));

        table.add_value(name, boost::any(5));
        BOOST_CHECK_EQUAL(table.size(), 1u);
        BOOST_REQUIRE(table.has_value(name));
        BOOST_CHECK_EQUAL(boost::any_cast<int>(table.value(name)), 5);
        BOOST_CHECK_EQUAL(table.scope_level(name), table.scope_level());

        BOOST_CHECK_EQUAL(table.scope_level(), SymbolTable::size_type(1));

        table.add_value(name, boost::any(6));
        BOOST_CHECK_EQUAL(table.size(), 1u);
        BOOST_REQUIRE(table.has_value(name));
        BOOST_CHECK_EQUAL(boost::any_cast<int>(table.value(name)), 6);
        BOOST_CHECK_EQUAL(table.scope_level(name), table.scope_level());

        // Should remove all definitions of 'a' in the current scope.
        table.pop_scope();
        BOOST_CHECK_EQUAL(table.size(), 0u);
        BOOST_REQUIRE(!table.has_value(name));

        BOOST_CHECK_EQUAL(table.scope_level(), SymbolTable::size_type(0));
    }

    // Add and remove two definitions, nested.
    {
        table.push_scope();
        BOOST_CHECK_EQUAL(table.scope_level(), SymbolTable::size_type(1));

        table.add_value(name, boost::any(5));
        BOOST_CHECK_EQUAL(table.size(), 1u);
        BOOST_REQUIRE(table.has_value(name));
        BOOST_CHECK_EQUAL(boost::any_cast<int>(table.value(name)), 5);

        table.push_scope();
        BOOST_CHECK_EQUAL(table.scope_level(), SymbolTable::size_type(2));

        table.add_value(name, boost::any(6));
        BOOST_CHECK_EQUAL(table.size(), 2u);
        BOOST_REQUIRE(table.has_value(name));
        BOOST_CHECK_EQUAL(boost::any_cast<int>(table.value(name)), 6);

        // Should reveal the first value.
        table.pop_scope();
        BOOST_CHECK_EQUAL(table.size(), 1u);
        BOOST_CHECK_EQUAL(table.scope_level(), SymbolTable::size_type(1));

        BOOST_REQUIRE(table.has_value(name));
        BOOST_CHECK_EQUAL(boost::any_cast<int>(table.value(name)), 5);
        BOOST_CHECK_EQUAL(table.scope_level(name), table.scope_level());

        table.pop_scope();
        BOOST_CHECK_EQUAL(table.size(), 0u);
        BOOST_REQUIRE(!table.has_value(name));

        BOOST_CHECK_EQUAL(table.scope_level(), SymbolTable::size_type(0));
    }
}


BOOST_AUTO_TEST_CASE(erase_value)
{
    using namespace fern;

    SymbolTable<int> table;

    table.push_scope();
    table.add_value("a", 5);
    table.add_value("b", 6);
    BOOST_CHECK(table.has_value("a"));
    BOOST_CHECK(table.has_value("b"));

    table.erase_value("a");
    BOOST_CHECK(!table.has_value("a"));
    BOOST_CHECK( table.has_value("b"));
}

BOOST_AUTO_TEST_SUITE_END()
