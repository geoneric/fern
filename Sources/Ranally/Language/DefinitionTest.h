#ifndef INCLUDED_RANALLY_LANGUAGE_DEFINITIONTEST
#define INCLUDED_RANALLY_LANGUAGE_DEFINITIONTEST



namespace boost {
  namespace unit_test {
    class test_suite;
  }
}



class DefinitionTest
{

public:

                   DefinitionTest           ();

  void             test                ();

  static boost::unit_test::test_suite* suite();

};

#endif
