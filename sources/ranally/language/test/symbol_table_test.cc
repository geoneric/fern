#define BOOST_TEST_MODULE ranally language
#include <boost/test/unit_test.hpp>
#include "ranally/language/name_vertex.h"
#include "ranally/language/symbol_table.h"


BOOST_AUTO_TEST_SUITE(symbol_table)


BOOST_AUTO_TEST_CASE(scoping)
{
    using namespace ranally;

    SymbolTable table;
    ranally::String name("a");
    BOOST_CHECK_EQUAL(table.scope_level(), SymbolTable::size_type(0));

    // Add and remove one value for 'a'.
    {
        table.push_scope();
        BOOST_CHECK_EQUAL(table.scope_level(), SymbolTable::size_type(1));

        std::unique_ptr<NameVertex> a(new NameVertex(name));
        table.add_value(name, a.get());
        BOOST_REQUIRE(table.has_value(name));
        BOOST_CHECK_EQUAL(table.scope_level(name), table.scope_level());

        table.pop_scope();
        BOOST_REQUIRE(!table.has_value(name));

        BOOST_CHECK_EQUAL(table.scope_level(), SymbolTable::size_type(0));
    }

    // Add and remove two definitions for 'a', not nested.
    {
        table.push_scope();
        BOOST_CHECK_EQUAL(table.scope_level(), SymbolTable::size_type(1));

        std::unique_ptr<NameVertex> a1(new NameVertex(name));
        table.add_value(name, a1.get());
        BOOST_REQUIRE(table.has_value(name));
        BOOST_CHECK_EQUAL(table.value(name), a1.get());
        BOOST_CHECK_EQUAL(table.scope_level(name), table.scope_level());

        BOOST_CHECK_EQUAL(table.scope_level(), SymbolTable::size_type(1));

        std::unique_ptr<NameVertex> a2(new NameVertex(name));
        table.add_value(name, a2.get());
        BOOST_REQUIRE(table.has_value(name));
        BOOST_CHECK_EQUAL(table.value(name), a2.get());
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

        std::unique_ptr<NameVertex> a1(new NameVertex(name));
        table.add_value(name, a1.get());
        BOOST_REQUIRE(table.has_value(name));

        table.push_scope();
        BOOST_CHECK_EQUAL(table.scope_level(), SymbolTable::size_type(2));

        std::unique_ptr<NameVertex> a2(new NameVertex(name));
        table.add_value(name, a2.get());
        BOOST_REQUIRE(table.has_value(name));

        // Should reveal the first value.
        table.pop_scope();
        BOOST_CHECK_EQUAL(table.scope_level(), SymbolTable::size_type(1));

        BOOST_REQUIRE(table.has_value(name));
        BOOST_CHECK_EQUAL(table.scope_level(name), table.scope_level());

        table.pop_scope();
        BOOST_REQUIRE(!table.has_value(name));

        BOOST_CHECK_EQUAL(table.scope_level(), SymbolTable::size_type(0));
    }
}

BOOST_AUTO_TEST_SUITE_END()
