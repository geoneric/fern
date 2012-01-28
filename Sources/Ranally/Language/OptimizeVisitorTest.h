#ifndef INCLUDED_RANALLY_LANGUAGE_PURIFYVISITORTEST
#define INCLUDED_RANALLY_LANGUAGE_PURIFYVISITORTEST



namespace boost {
  namespace unit_test {
    class test_suite;
  }
}



class PurifyVisitorTest
{

public:

                   PurifyVisitorTest   ();

  void             test                ();

  static boost::unit_test::test_suite* suite();

};

#endif
