#ifndef INCLUDED_RANALLY_SCRIPTVERTEXTEST
#define INCLUDED_RANALLY_SCRIPTVERTEXTEST



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
