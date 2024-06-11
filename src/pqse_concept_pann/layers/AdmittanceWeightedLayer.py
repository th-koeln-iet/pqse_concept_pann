from pqse_concept_pann.layers import AdmittanceLayer
from pqse_concept_pann.tools.scaling import DimMinMaxScaler
from pqse_concept_pann.tools.scaling import redistribute_values


class AdmittanceWeightedLayer(AdmittanceLayer):
    def __init__(self, y_mats, mask_indices=None, activation=None, redistribution_factor=50000,
                 data_format='channels_last', **kwargs):
        super(AdmittanceWeightedLayer, self).__init__(y_mats, mask_indices, activation, data_format=data_format,
                                                      **kwargs)
        self.redistribution_factor = redistribution_factor

    def build(self, input_shape):
        # normalize the adjacency matrix
        scaler = DimMinMaxScaler()
        self.y_mats = scaler.fit_transform(self.y_mats, self.y_node_axes)  # scale over axes corresponding to nodes
        # This results in some values being very close to 1 and others being very close to 0
        # For our neural network, we need a more even distribution
        self.y_mats = redistribute_values(self.y_mats, k=self.redistribution_factor,
                                          axis=self.y_node_axes)
        super().build(input_shape)
