#ifndef INCLUDED_RANALLY_SYMBOLTABLETEST
#define INCLUDED_RANALLY_SYMBOLTABLETEST



namespace boost {
  namespace unit_test {
    class test_suite;
  }
}



class SymbolTableTest
{

private:

public:

                   SymbolTableTest           ();

  void             test                ();

  static boost::unit_test::test_suite* suite();

};

#endif
