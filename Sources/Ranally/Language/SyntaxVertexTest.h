#ifndef INCLUDED_RANALLY_LANGUAGE_SYNTAXVERTEXTEST
#define INCLUDED_RANALLY_LANGUAGE_SYNTAXVERTEXTEST



namespace boost {
  namespace unit_test {
    class test_suite;
  }
}



class SyntaxVertexTest
{

private:

public:

                   SyntaxVertexTest    ();

  void             test                ();

  static boost::unit_test::test_suite* suite();

};

#endif
