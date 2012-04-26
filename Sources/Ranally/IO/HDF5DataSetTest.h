#ifndef INCLUDED_RANALLY_IO_HDF5DATASETTEST
#define INCLUDED_RANALLY_IO_HDF5DATASETTEST



namespace boost {
  namespace unit_test {
    class test_suite;
  }
}



class HDF5DataSetTest
{

public:

                   HDF5DataSetTest     ();

  void             test                ();

  static boost::unit_test::test_suite* suite();

private:

};

#endif
