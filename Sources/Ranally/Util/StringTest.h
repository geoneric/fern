#pragma once


namespace boost {
    namespace unit_test {
        class test_suite;
    }
}


class StringTest
{

public:

                   StringTest          ();

  void             testEncodeInUTF8    ();

  static boost::unit_test::test_suite* suite();

private:

};
