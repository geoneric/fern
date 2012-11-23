#pragma once
#include "Ranally/IO/dataset.h"


class OGRDataSource;

namespace ranally {

class OGRFeatureLayer;

//! Class representing an OGR data set with feature geometries and attributes.
/*!
  The OGRDataset class is comparable with the OGRDataSource class as defined
  in OGR's API. An OGRLayer as used in OGR is what we call a Feature in
  Ranally.

  \sa        .
*/
class OGRDataset:
    public Dataset
{

    friend class OGRDatasetTest;

public:

                   OGRDataset          (String const& name,
                                        OGRDataSource* dataSource);

                   ~OGRDataset         ();

    size_t         nrFeatures          () const;

    Feature*       feature             (size_t i) const;

    Feature*       feature             (String const& name) const;

    void           addFeature          (Feature const& feature);

    void           copy                (Dataset const& dataSet);

    bool           exists              (String const& name) const;

    void           remove              (String const& name);

private:

    OGRDataSource* _dataSource;

    void           copy                (Feature const& feature);

    Feature*       feature             (OGRFeatureLayer const& layer) const;

    template<class Feature>
    void           add                 (Feature const& feature);

};

} // namespace ranally
