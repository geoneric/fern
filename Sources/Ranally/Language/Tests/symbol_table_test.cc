#define BOOST_TEST_MODULE ranally language
#include <boost/test/included/unit_test.hpp>
#include "Ranally/Language/SymbolTable.h"
#include "Ranally/Language/NameVertex.h"


BOOST_AUTO_TEST_SUITE(symbol_table)


BOOST_AUTO_TEST_CASE(scoping)
{
    using namespace ranally;

    SymbolTable table;
    ranally::String name("a");
    BOOST_CHECK_EQUAL(table.scopeLevel(), SymbolTable::size_type(0));

    // Add and remove one definition for 'a'.
    {
        table.pushScope();
        BOOST_CHECK_EQUAL(table.scopeLevel(), SymbolTable::size_type(1));

        std::unique_ptr<NameVertex> a(new NameVertex(name));
        table.addDefinition(a.get());
        BOOST_REQUIRE(table.hasDefinition(name));
        BOOST_CHECK_EQUAL(table.scopeLevel(name), table.scopeLevel());

        table.popScope();
        BOOST_REQUIRE(!table.hasDefinition(name));

        BOOST_CHECK_EQUAL(table.scopeLevel(), SymbolTable::size_type(0));
    }

    // Add and remove two definitions for 'a', not nested.
    {
        table.pushScope();
        BOOST_CHECK_EQUAL(table.scopeLevel(), SymbolTable::size_type(1));

        std::unique_ptr<NameVertex> a1(new NameVertex(name));
        table.addDefinition(a1.get());
        BOOST_REQUIRE(table.hasDefinition(name));
        BOOST_CHECK_EQUAL(table.definition(name), a1.get());
        BOOST_CHECK_EQUAL(table.scopeLevel(name), table.scopeLevel());

        BOOST_CHECK_EQUAL(table.scopeLevel(), SymbolTable::size_type(1));

        std::unique_ptr<NameVertex> a2(new NameVertex(name));
        table.addDefinition(a2.get());
        BOOST_REQUIRE(table.hasDefinition(name));
        BOOST_CHECK_EQUAL(table.definition(name), a2.get());
        BOOST_CHECK_EQUAL(table.scopeLevel(name), table.scopeLevel());

        // Should remove all definitions of 'a' in the current scope.
        table.popScope();
        BOOST_REQUIRE(!table.hasDefinition(name));

        BOOST_CHECK_EQUAL(table.scopeLevel(), SymbolTable::size_type(0));
    }

    // Add and remove two definitions, nested.
    {
        table.pushScope();
        BOOST_CHECK_EQUAL(table.scopeLevel(), SymbolTable::size_type(1));

        std::unique_ptr<NameVertex> a1(new NameVertex(name));
        table.addDefinition(a1.get());
        BOOST_REQUIRE(table.hasDefinition(name));

        table.pushScope();
        BOOST_CHECK_EQUAL(table.scopeLevel(), SymbolTable::size_type(2));

        std::unique_ptr<NameVertex> a2(new NameVertex(name));
        table.addDefinition(a2.get());
        BOOST_REQUIRE(table.hasDefinition(name));

        // Should reveal the first definition.
        table.popScope();
        BOOST_CHECK_EQUAL(table.scopeLevel(), SymbolTable::size_type(1));

        BOOST_REQUIRE(table.hasDefinition(name));
        BOOST_CHECK_EQUAL(table.scopeLevel(name), table.scopeLevel());

        table.popScope();
        BOOST_REQUIRE(!table.hasDefinition(name));

        BOOST_CHECK_EQUAL(table.scopeLevel(), SymbolTable::size_type(0));
    }
}

BOOST_AUTO_TEST_SUITE_END()
