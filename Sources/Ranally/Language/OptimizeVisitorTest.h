#ifndef INCLUDED_RANALLY_LANGUAGE_OPTIMIZEVISITORTEST
#define INCLUDED_RANALLY_LANGUAGE_OPTIMIZEVISITORTEST



namespace boost {
  namespace unit_test {
    class test_suite;
  }
}



class OptimizeVisitorTest
{

public:

                   OptimizeVisitorTest ();

  void             test                ();

  static boost::unit_test::test_suite* suite();

};

#endif
