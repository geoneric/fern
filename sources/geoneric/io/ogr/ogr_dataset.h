#pragma once
#include "geoneric/io/core/dataset.h"


class OGRDataSource;

namespace geoneric {

class OGRFeatureLayer;

//! Class representing an OGR data set with feature geometries and attributes.
/*!
  The OGRDataset class is comparable with the OGRDataSource class as defined
  in OGR's API. An OGRLayer as used in OGR is what we call a Feature in
  Geoneric.

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

    size_t         nr_features         () const;

    Feature*       feature             (size_t i) const;

    Feature*       feature             (String const& name) const;

    void           add_feature         (Feature const& feature);

    void           copy                (Dataset const& dataset);

    bool           exists              (String const& name) const;

    void           remove              (String const& name);

private:

    OGRDataSource* _data_source;

    void           copy                (Feature const& feature);

    Feature*       feature             (OGRFeatureLayer const& layer) const;

    template<class Feature>
    void           add                 (Feature const& feature);

};

} // namespace geoneric
