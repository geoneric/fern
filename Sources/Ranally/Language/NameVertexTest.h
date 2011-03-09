#ifndef INCLUDED_RANALLY_LANGUAGE_NAMEVERTEXTEST
#define INCLUDED_RANALLY_LANGUAGE_NAMEVERTEXTEST



namespace boost {
  namespace unit_test {
    class test_suite;
  }
}



class NameVertexTest
{

public:

                   NameVertexTest      ();

  void             test                ();

  static boost::unit_test::test_suite* suite();

};

#endif
