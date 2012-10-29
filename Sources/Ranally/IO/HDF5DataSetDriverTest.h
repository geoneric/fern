#pragma once


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
