#ifndef INCLUDED_RANALLY_SYNTAXTREETEST
#define INCLUDED_RANALLY_SYNTAXTREETEST



namespace boost {
  namespace unit_test {
    class test_suite;
  }
}



class SyntaxTreeTest
{

private:

public:

                   SyntaxTreeTest      ();

  void             test                ();

  static boost::unit_test::test_suite* suite();

};

#endif
