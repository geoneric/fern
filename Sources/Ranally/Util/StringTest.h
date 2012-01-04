#ifndef INCLUDED_RANALLY_UTIL_STRINGTEST
#define INCLUDED_RANALLY_UTIL_STRINGTEST



namespace boost {
  namespace unit_test {
    class test_suite;
  }
}



class StringTest
{

private:

public:

                   StringTest          ();

  void             testEncodeInUTF8    ();

  static boost::unit_test::test_suite* suite();

};

#endif
