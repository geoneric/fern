#ifndef INCLUDED_RANALLY_LANGUAGE_SYMBOLTABLETEST
#define INCLUDED_RANALLY_LANGUAGE_SYMBOLTABLETEST



namespace boost {
  namespace unit_test {
    class test_suite;
  }
}



class SymbolTableTest
{

private:

public:

                   SymbolTableTest     ();

  void             testScoping         ();

  static boost::unit_test::test_suite* suite();

};

#endif
