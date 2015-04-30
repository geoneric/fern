// -----------------------------------------------------------------------------
// Fern © Geoneric
//
// This file is part of Geoneric Fern which is available under the terms of
// the GNU General Public License (GPL), version 2. If you do not want to
// be bound by the terms of the GPL, you may purchase a proprietary license
// from Geoneric (http://www.geoneric.eu/contact).
// -----------------------------------------------------------------------------
#pragma once
#include "fern/language/io/core/dataset.h"


class OGRDataSource;

namespace fern {

class OGRFeatureLayer;

//! Class representing an OGR data set with feature geometries and attributes.
/*!
  The OGRDataset class is comparable with the OGRDataSource class as defined
  in OGR's API. An OGRLayer as used in OGR is what we call a Feature in
  Fern.

  \sa        .
*/
class OGRDataset:
    public Dataset
{

    friend class OGRDatasetTest;

public:

                   OGRDataset          (std::string const& name,
                                        OGRDataSource* dataSource);

                   ~OGRDataset         ();

    size_t         nr_features         () const;

    Feature*       feature             (size_t i) const;

    Feature*       feature             (std::string const& name) const;

    void           add_feature         (Feature const& feature);

    void           copy                (Dataset const& dataset);

    bool           exists              (std::string const& name) const;

    void           remove              (std::string const& name);

private:

    OGRDataSource* _data_source;

    void           copy                (Feature const& feature);

    Feature*       feature             (OGRFeatureLayer const& layer) const;

    template<class Feature>
    void           add                 (Feature const& feature);

};

} // namespace fern
