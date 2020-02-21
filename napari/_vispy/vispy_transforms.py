import numpy as np
from vispy.visuals.transforms import STTransform, ChainTransform

from ..layers.transforms import TransformChain


class VispyTransformChain(ChainTransform):
    """Class containing an ordered sequence of vispy transforms.

    Parameters
    ----------
    napari_transform_chain : napari.layers.transforms.TransformChain
        Transform chain for an ordered sequence of napari transforms.

    Attributes
    ----------
    changed : #TODO
        #TODO
    scale : ndarray, shape(4)
        Combined scale of all chained transforms, vispy format (X,Y,Z,T).
    simplified_transform : vispy.visuals.transforms.ChainTransform
        Simplified ChainTransform where transforms have been combined into one.
        Used to reduce compute time, since a single matrix operation is faster.
    transform_chain : vispy.visuals.transforms.ChainTransform
        Transform chain for an ordered sequence of vispy transforms.
    translate : ndarray, shape(4)
        Combined tranlation of all chained transforms, vispy format (X,Y,Z,T).
    """

    def __init__(self, napari_transform_chain=TransformChain()):
        self.napari_transform_chain = napari_transform_chain

        self.transform_chain = self._vispy_transform_chain()
        self._calculate_attributes()

        self.changed = self.transform_chain.changed

        self.napari_transform_chain.events.added.connect(self._add)
        self.napari_transform_chain.events.removed.connect(self._remove)
        self.napari_transform_chain.events.reordered.connect(self._reorder)
        self.napari_transform_chain.events.changed.connect(self._changed)

    def _vispy_transform_chain(self):
        """Builds vispy ChainTransform from a napari TransformChain."""
        vispy_transforms = []
        for t in self.napari_transform_chain:
            # From napari format (t, z, y, x)
            # to the vispy format (x, y, z, ?)
            translate = np.flip(t.translate)  # vispy ordered array
            scale = np.flip(t.scale)  # vispy ordered array
            vispy_transforms.append(
                STTransform(scale=scale, translate=translate)
            )
        vispy_transform_chain = ChainTransform(vispy_transforms)
        self.transform_chain = vispy_transform_chain
        self.__dict__.update(vispy_transform_chain.__dict__)
        self._calculate_attributes()
        return vispy_transform_chain

    def _calculate_attributes(self):
        """Simplified transformation matrix, scale, and transle attributes.

        Combines all transforms in the VispyTransformChain.
        """
        if self.transform_chain.transforms != []:
            for idx, t in enumerate(self.transform_chain.transforms):
                if idx == 0:
                    tmp_scale = t.scale
                    tmp_translate = t.translate
                else:
                    tmp_scale = tmp_scale * t.scale
                    tmp_translate = (tmp_translate * t.scale) + t.translate
            self.scale = tmp_scale
            self.translate = tmp_translate
            self.simplified_transform = self.transform_chain.simplified
        else:
            self.scale = None
            self.translate = None
            self.simplified_transform = self.transform_chain.simplified

    def _add(self, event):
        """Insert vispy transform `event.item` at index `event.index`."""
        transform_index = event.index
        # Convert from napari format (t, z, y, x) to vispy format (x, y, z, ?)
        scale = np.flip(event.item.scale)  # vispy ordered array
        translate = np.flip(event.item.translate)  # vispy ordered array
        transform = STTransform(scale=scale, translate=translate)
        self.transform_chain.transforms.insert(transform_index, transform)
        self._calculate_attributes()

    def _remove(self, event):
        """Remove vispy transform at index `event.index`."""
        self.transform_chain.transforms.pop(event.index)
        self._calculate_attributes()

    def _reorder(self, event=None):
        raise NotImplementedError()

    def _changed(self, event=None):
        raise NotImplementedError()
