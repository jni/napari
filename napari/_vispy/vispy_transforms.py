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
    scale : ndarray, shape(4)
        Combined scale of all chained transforms, vispy format (X,Y,Z,T).
    simplified : vispy.visuals.transforms.ChainTransform
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

    @property
    def scale(self):
        """Combined scale of all elements in the VispyChainTransform.

        You should not directly modify the scale property of a
        VispyChainTransform. Instead you should append or insert another
        transform to the napari TransformChain, which will update this
        property using event emitters.

        Returns
        -------
        ndarray, shape (4,)
            Combined scale of all elements in the VispyChainTransform.
            Vispy format expects (X,Y,Z,T)
        """
        self._calculate_attributes()
        return self._scale

    @property
    def translate(self):
        """Combined translation of all elements in the VispyChainTransform.

        You should not directly modify the translate property of a
        VispyChainTransform. Instead you should append or insert another
        transform to the napari TransformChain, which will update this
        property using event emitters.

        Returns
        -------
        ndarray, shape (4,)
            Combined translation of all elements in the VispyChainTransform.
            Vispy format expects (X,Y,Z,T)
        """
        self._calculate_attributes()
        return self._translate

    @property
    def simplified(self):
        return self._simplified

    def _vispy_transform_chain(self):
        """Builds vispy ChainTransform from a napari TransformChain."""
        vispy_transforms = []
        for t in self.napari_transform_chain:
            # From napari format (t, z, y, x)
            # to the vispy format (x, y, z, ?)
            _translate = np.flip(t.translate)  # vispy ordered array
            _scale = np.flip(t.scale)  # vispy ordered array
            vispy_transforms.append(
                STTransform(scale=_scale, translate=_translate)
            )
        vispy_transform_chain = ChainTransform(vispy_transforms)
        self.transform_chain = vispy_transform_chain
        self.__dict__.update(vispy_transform_chain.__dict__)
        return vispy_transform_chain

    def _calculate_attributes(self):
        """Simplified transformation matrix, scale, and transle attributes.

        Combines all transforms in the VispyTransformChain.
        """
        if self.transform_chain.transforms != []:
            for idx, t in enumerate(self.transform_chain.transforms):
                if t.scale is None:
                    t.scale = np.array([1.0])
                if t.translate is None:
                    t.translate = np.array([0.0])
                if idx == 0:
                    tmp_scale = t.scale
                    tmp_translate = t.translate
                else:
                    tmp_scale = tmp_scale * t.scale
                    tmp_translate = (tmp_translate * t.scale) + t.translate
            self._scale = tmp_scale
            self._translate = tmp_translate
            self._simplified = self.transform_chain.simplified
        else:
            self._scale = np.array([1.0, 1.0, 1.0, 1.0])
            self._translate = np.array([0.0, 0.0, 0.0, 0.0])
            self._simplified = self.transform_chain.simplified

    def _add(self, event):
        """Insert vispy transform `event.item` at index `event.index`."""
        transform_index = event.index
        # Convert from napari format (t, z, y, x) to vispy format (x, y, z, ?)
        _scale = np.flip(event.item.scale)  # vispy ordered array
        _translate = np.flip(event.item.translate)  # vispy ordered array
        _transform = STTransform(scale=_scale, translate=_translate)
        self.transform_chain.transforms.insert(transform_index, _transform)
        self._calculate_attributes()

    def _remove(self, event):
        """Remove vispy transform at index `event.index`."""
        self.transform_chain.transforms.pop(event.index)

    def _reorder(self, event=None):
        raise NotImplementedError()
