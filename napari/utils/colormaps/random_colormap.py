from typing import Union

import numpy as np

from ...utils.events import EventedModel
from ..translations import trans
from .standardize_color import transform_color


class LabelsRandomColormap(EventedModel):
    """Colormap that converts integer features to random colors.

    Parameters
    ----------
    colormap : np.ndarray
        The colors in the colormap. Labels will be randomly or pseudorandomly
        mapped to these values.
    background : np.ndarray
        The value 0 will be mapped to this color.
    """

    colormap: np.ndarray

    def map(self, color_properties: Union[list, np.ndarray]) -> np.ndarray:
        """Map an array of values to an array of colors
        Parameters
        ----------
        color_properties : Union[list, np.ndarray]
            The property values to be converted to colors.
        Returns
        -------
        colors : np.ndarray
            An Nx4 color array where N is the number of property values provided.
        """
        if isinstance(color_properties, (list, np.ndarray)):
            color_properties = np.asarray(color_properties)
        else:
            color_properties = np.asarray([color_properties])

        # add properties if they are not in the colormap
        color_cycle_keys = [*self.colormap]
        props_in_map = np.in1d(color_properties, color_cycle_keys)
        if not np.all(props_in_map):
            new_prop_values = color_properties[np.logical_not(props_in_map)]
            indices_to_add = np.unique(new_prop_values, return_index=True)[1]
            props_to_add = [
                new_prop_values[index] for index in sorted(indices_to_add)
            ]
            for prop in props_to_add:
                new_color = next(self.fallback_color.cycle)
                self.colormap[prop] = np.squeeze(transform_color(new_color))
        # map the colors
        colors = np.array([self.colormap[x] for x in color_properties])
        return colors

    @classmethod
    def from_dict(cls, params: dict):
        if ('colormap' in params) or ('fallback_color' in params):
            if 'colormap' in params:
                colormap = {
                    k: transform_color(v)[0]
                    for k, v in params['colormap'].items()
                }
            else:
                colormap = {}
            if 'fallback_color' in params:
                fallback_color = params['fallback_color']
            else:
                fallback_color = 'white'
        else:
            colormap = {k: transform_color(v)[0] for k, v in params.items()}
            fallback_color = 'white'

        return cls(colormap=colormap, fallback_color=fallback_color)

    @classmethod
    def __get_validators__(cls):
        yield cls.validate_type

    @classmethod
    def validate_type(cls, val):
        if isinstance(val, cls):
            return val
        if isinstance(val, list) or isinstance(val, np.ndarray):
            return cls.from_array(val)
        elif isinstance(val, dict):
            return cls.from_dict(val)
        else:
            raise TypeError(
                trans._(
                    'colormap should be an array or dict',
                    deferred=True,
                )
            )

    def __eq__(self, other):
        return isinstance(other, LabelsRandomColormap) and np.all(
            self.colormap == other.colormap
        )
