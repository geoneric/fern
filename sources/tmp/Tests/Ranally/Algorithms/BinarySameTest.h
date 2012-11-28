#ifndef INCLUDED_BINARYSAMETEST
#define INCLUDED_BINARYSAMETEST



namespace boost {
  namespace unit_test {
    class test_suite;
  }
}



class BinarySameTest
{

private:

public:

                   BinarySameTest      ();

  void             test                ();

  static boost::unit_test::test_suite* suite();

};

#endif
