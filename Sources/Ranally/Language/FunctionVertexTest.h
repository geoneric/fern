#ifndef INCLUDED_RANALLY_LANGUAGE_FUNCTIONVERTEXTEST
#define INCLUDED_RANALLY_LANGUAGE_FUNCTIONVERTEXTEST



namespace boost {
  namespace unit_test {
    class test_suite;
  }
}



class FunctionVertexTest
{

public:

                   FunctionVertexTest  ();

  void             test                ();

  static boost::unit_test::test_suite* suite();

};

#endif
