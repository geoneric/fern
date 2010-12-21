#ifndef INCLUDED_RANALLY_PURIFYVISITORTEST
#define INCLUDED_RANALLY_PURIFYVISITORTEST



namespace boost {
  namespace unit_test {
    class test_suite;
  }
}



class PurifyVisitorTest
{

private:

public:

                   PurifyVisitorTest   ();

  void             test                ();

  static boost::unit_test::test_suite* suite();

};

#endif
