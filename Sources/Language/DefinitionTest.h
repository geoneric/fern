#ifndef INCLUDED_RANALLY_DEFINITIONTEST
#define INCLUDED_RANALLY_DEFINITIONTEST



namespace boost {
  namespace unit_test {
    class test_suite;
  }
}



class DefinitionTest
{

private:

public:

                   DefinitionTest           ();

  void             test                ();

  static boost::unit_test::test_suite* suite();

};

#endif
