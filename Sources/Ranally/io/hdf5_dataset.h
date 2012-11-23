#pragma once
#include <memory>
#include "ranally/io/dataset.h"


namespace H5 {
    class H5File;
} // namespace H5

namespace ranally {

//! Class for HDF5 data sets.
/*!
  HDF5 data sets represent data sets stored using the HDF5 library.

  \sa        HDF5DatasetDriver
*/
class HDF5Dataset:
    public Dataset
{

    friend class HDF5DatasetTest;

public:

                   HDF5Dataset         (String const& name,
                                        H5::H5File* file);

                   ~HDF5Dataset        ();

    size_t         nrFeatures          () const;

    Feature*       feature             (size_t i) const;

    Feature*       feature             (String const& name) const;

    void           addFeature          (Feature const& feature);

    void           copy                (Dataset const& dataSet);

    bool           exists              (String const& name) const;

    void           remove              (String const& name);

private:

    std::unique_ptr<H5::H5File> _file;

    void           copy                (Feature const& feature);

    template<class Feature>
    void           add                 (Feature const& feature);

};

} // namespace ranally
