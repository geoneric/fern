#pragma once


namespace boost {
    namespace unit_test {
        class test_suite;
    }
}


class PlusTest
{

public:

                   PlusTest            ();

  void             testArgumentAndResultTypes();

  static boost::unit_test::test_suite* suite();

private:

};
