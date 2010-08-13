#ifndef INCLUDED_RANALLY_ALGEBRAPARSERTEST
#define INCLUDED_RANALLY_ALGEBRAPARSERTEST



namespace boost {
  namespace unit_test {
    class test_suite;
  }
}



class AlgebraParserTest
{

private:

public:

                   AlgebraParserTest   ();

  void             testParseString     ();

  void             testParseFile       ();

  static boost::unit_test::test_suite* suite();

};

#endif
