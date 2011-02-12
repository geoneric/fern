#ifndef INCLUDED_RANALLY_LANGUAGE_SCRIPTVERTEXTEST
#define INCLUDED_RANALLY_LANGUAGE_SCRIPTVERTEXTEST



namespace boost {
  namespace unit_test {
    class test_suite;
  }
}



class ScriptVertexTest
{

private:

public:

                   ScriptVertexTest    ();

  void             test                ();

  static boost::unit_test::test_suite* suite();

};

#endif
