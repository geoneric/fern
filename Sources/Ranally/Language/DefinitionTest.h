#pragma once


namespace boost {
    namespace unit_test {
        class test_suite;
    }
}


class DefinitionTest
{

public:

                   DefinitionTest      ();

    void           test                ();

    static boost::unit_test::test_suite* suite();

};
