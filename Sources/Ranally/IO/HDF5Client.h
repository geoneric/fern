#pragma once
#include <boost/noncopyable.hpp>



namespace ranally {
namespace io {

//! Class that encapsulates the configuration of the HDF5 library for the HDF5 client.
/*!
  In case an error occurs in one of HDF5's C++ API calls, an exception is
  thrown, instead of a message being printed.
*/
class HDF5Client:
  private boost::noncopyable
{

  friend class HDF5ClientTest;

public:

  virtual          ~HDF5Client         ();

protected:

                   HDF5Client          ();

private:

};

} // namespace io
} // namespace ranally
