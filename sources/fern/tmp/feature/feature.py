class Feature(object):

    def __init__(self,
            domain):
        self.domain = domain
        self.attributes = []

    def add_attribute(self,
            attribute):
        self.attributes.append(attribute)


class Domain(object):

    def __init__(self,
            temporal_domain=None,
            spatial_domain=None):
        self.temporal_domain = temporal_domain
        self.spatial_domain = spatial_domain


class Attribute(object):

    def __init__(self,
            value,
            name=None):
        self.value = value
        self.name = name


scalar_feature = Feature(Domain())
scalar_feature.add_attribute(Attribute(5))

scalar_attribute = Attribute(5)


print scalar_feature
