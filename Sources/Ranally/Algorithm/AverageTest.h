#ifndef INCLUDED_RANALLY_ALGORITHM_AVERAGETEST
#define INCLUDED_RANALLY_ALGORITHM_AVERAGETEST



namespace boost {
  namespace unit_test {
    class test_suite;
  }
}



class AverageTest
{

public:

                   AverageTest         ();

  void             test                ();

  static boost::unit_test::test_suite* suite();

private:

};

#endif
