#define BOOST_TEST_MODULE ranally core
#include <boost/any.hpp>
#include <boost/test/unit_test.hpp>
#include "ranally/core/symbol_table.h"


BOOST_AUTO_TEST_SUITE(symbol_table)


BOOST_AUTO_TEST_CASE(scoping)
{
    using namespace ranally;

    typedef SymbolTable<boost::any> SymbolTable;

    SymbolTable table;
    String name("a");
    BOOST_CHECK_EQUAL(table.scope_level(), SymbolTable::size_type(0));

    // Add and remove one value for 'a'.
    {
        table.push_scope();
        BOOST_CHECK_EQUAL(table.scope_level(), SymbolTable::size_type(1));

        table.add_value(name, boost::any(5));
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
        BOOST_REQUIRE(table.has_value(name));
        BOOST_CHECK_EQUAL(boost::any_cast<int>(table.value(name)), 5);
        BOOST_CHECK_EQUAL(table.scope_level(name), table.scope_level());

        BOOST_CHECK_EQUAL(table.scope_level(), SymbolTable::size_type(1));

        table.add_value(name, boost::any(6));
        BOOST_REQUIRE(table.has_value(name));
        BOOST_CHECK_EQUAL(boost::any_cast<int>(table.value(name)), 6);
        BOOST_CHECK_EQUAL(table.scope_level(name), table.scope_level());

        // Should remove all definitions of 'a' in the current scope.
        table.pop_scope();
        BOOST_REQUIRE(!table.has_value(name));

        BOOST_CHECK_EQUAL(table.scope_level(), SymbolTable::size_type(0));
    }

    // Add and remove two definitions, nested.
    {
        table.push_scope();
        BOOST_CHECK_EQUAL(table.scope_level(), SymbolTable::size_type(1));

        table.add_value(name, boost::any(5));
        BOOST_REQUIRE(table.has_value(name));
        BOOST_CHECK_EQUAL(boost::any_cast<int>(table.value(name)), 5);

        table.push_scope();
        BOOST_CHECK_EQUAL(table.scope_level(), SymbolTable::size_type(2));

        table.add_value(name, boost::any(6));
        BOOST_REQUIRE(table.has_value(name));
        BOOST_CHECK_EQUAL(boost::any_cast<int>(table.value(name)), 6);

        // Should reveal the first value.
        table.pop_scope();
        BOOST_CHECK_EQUAL(table.scope_level(), SymbolTable::size_type(1));

        BOOST_REQUIRE(table.has_value(name));
        BOOST_CHECK_EQUAL(boost::any_cast<int>(table.value(name)), 5);
        BOOST_CHECK_EQUAL(table.scope_level(name), table.scope_level());

        table.pop_scope();
        BOOST_REQUIRE(!table.has_value(name));

        BOOST_CHECK_EQUAL(table.scope_level(), SymbolTable::size_type(0));
    }
}

BOOST_AUTO_TEST_SUITE_END()
