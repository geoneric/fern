#ifndef INCLUDED_RANALLY_LANGUAGE_STRINGVERTEXTEST
#define INCLUDED_RANALLY_LANGUAGE_STRINGVERTEXTEST



namespace boost {
  namespace unit_test {
    class test_suite;
  }
}



class StringVertexTest
{

public:

                   StringVertexTest    ();

  void             test                ();

  static boost::unit_test::test_suite* suite();

};

#endif
