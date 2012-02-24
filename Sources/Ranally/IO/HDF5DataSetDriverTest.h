#ifndef INCLUDED_RANALLY_IO_HDF5DATASETDRIVERTEST
#define INCLUDED_RANALLY_IO_HDF5DATASETDRIVERTEST



namespace boost {
  namespace unit_test {
    class test_suite;
  }
}



class HDF5DataSetDriverTest
{

public:

                   HDF5DataSetDriverTest();

                   ~HDF5DataSetDriverTest();

  void             testExists          ();

  void             testCreate          ();

  void             testRemove          ();

  void             testOpen            ();

  static boost::unit_test::test_suite* suite();

private:

  void             removeTestFiles     ();

};

#endif
