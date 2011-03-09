#ifndef INCLUDED_RANALLY_LANGUAGE_IFVERTEXTEST
#define INCLUDED_RANALLY_LANGUAGE_IFVERTEXTEST



namespace boost {
  namespace unit_test {
    class test_suite;
  }
}



class IfVertexTest
{

public:

                   IfVertexTest        ();

  void             test                ();

  static boost::unit_test::test_suite* suite();

};

#endif
