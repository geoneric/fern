#pragma once


namespace boost {
    namespace unit_test {
        class test_suite;
    }
}


class SymbolTableTest
{

public:

                   SymbolTableTest     ();

    void           testScoping         ();

    static boost::unit_test::test_suite* suite();

};
