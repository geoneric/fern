#pragma once


namespace boost {
    namespace unit_test {
        class test_suite;
    }
}


class ValidateVisitorTest
{

public:

                   ValidateVisitorTest ();

    void           test                ();

    static boost::unit_test::test_suite* suite();

private:

};
