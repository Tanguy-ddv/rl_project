from .custom_hopper import CustomHopper
import numpy as np

ADR = 'ADRHopper-v0'
UDR = 'UDRHopper-v0'
NDR = 'GDRHopper-v0'

class UDRHopper(CustomHopper):

    def __init__(self):
        super().__init__('source')

        self.params = self.get_parameters()
    
    def modify_parameters(self, upperbound: float = None, lowerbound: float = None):
        """
        Sample new parameters with a uniform distribution.
        """
        if lowerbound is not None:
            lower = np.ones_like(self.params)*lowerbound
        else:
            lower = np.zeros_like(self.params)
        if upper is not None:
            upper = np.ones_like(self.params)*upperbound
        else:
            upper = 2*self.params
        self.set_parameters(np.random.uniform(lower, upper))

class GDRHopper(CustomHopper):
    """A gaussian domain randomization hopper."""

    def __init__(self):
        super().__init__('source')

        self.params = self.get_parameters()

    def modify_parameters(self, sigma: float = 0.2):
        """
        Sample new parameters with a normal distribution (means=initial_values, std=initial_values*sigma) .
        """
        means = self.params
        stds = self.params*sigma
        self.set_parameters(np.random.normal(means, stds))

class ADRHopper(CustomHopper):
    """An active domain randomization hopper."""
        
    def __init__(self):
        super().__init__('source')

    def modify_parameters(self, params: np.ndarray):
        """
        Set the new parameters with the output of a particle.
        """
        self.set_parameters(params)