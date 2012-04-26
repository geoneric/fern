#ifndef INCLUDED_RANALLY_IO_OGRDATASETDRIVERTEST
#define INCLUDED_RANALLY_IO_OGRDATASETDRIVERTEST



namespace boost {
  namespace unit_test {
    class test_suite;
  }
}



class OGRDataSetDriverTest
{

public:

                   OGRDataSetDriverTest();

                   ~OGRDataSetDriverTest();

  void             test                ();

  void             testExists          ();

  void             testCreate          ();

  void             testRemove          ();

  void             testOpen            ();

  static boost::unit_test::test_suite* suite();

private:

  void             removeTestFiles     ();

};

#endif
