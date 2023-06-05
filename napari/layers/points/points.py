import numbers
import warnings
from abc import abstractmethod
from copy import copy, deepcopy
from itertools import cycle
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike
from psygnal.containers import Selection
from scipy.stats import gmean

from napari.layers.base import Layer, no_op
from napari.layers.base._base_mouse_bindings import (
    highlight_box_handles,
    transform_with_box,
)
from napari.layers.points._points_constants import Mode, Shading
from napari.layers.points._points_mouse_bindings import add, highlight, select
from napari.layers.points._points_utils import (
    _create_box_from_corners_3d,
    coerce_symbols,
    create_box,
    fix_data_points,
    points_to_squares,
)
from napari.layers.points._slice import _PointSliceRequest, _PointSliceResponse
from napari.layers.utils._color_manager_constants import ColorMode
from napari.layers.utils._slice_input import _SliceInput
from napari.layers.utils.color_manager import ColorManager
from napari.layers.utils.color_transformations import ColorType
from napari.layers.utils.interactivity_utils import (
    displayed_plane_from_nd_line_segment,
)
from napari.layers.utils.layer_utils import (
    _features_to_properties,
    _FeatureTable,
    _unique_element,
)
from napari.layers.utils.text_manager import TextManager
from napari.utils.colormaps import Colormap, ValidColormapArg
from napari.utils.colormaps.standardize_color import hex_to_name, rgb_to_hex
from napari.utils.events import Event
from napari.utils.events.custom_types import Array
from napari.utils.geometry import project_points_onto_plane, rotate_points
from napari.utils.status_messages import generate_layer_coords_status
from napari.utils.transforms import Affine
from napari.utils.translations import trans

DEFAULT_COLOR_CYCLE = np.array([[1, 0, 1, 1], [0, 1, 0, 1]])


class _BasePoints(Layer):
    """
    Implements the basic functionality of spatially distributed coordinates.
    Used by to display points and graph nodes.

    TODO: update documentation and typing

    Parameters
    ----------
    data : array (N, D)
        Coordinates for N points in D dimensions.
    ndim : int
        Number of dimensions for shapes. When data is not None, ndim must be D.
        An empty points layer can be instantiated with arbitrary ndim.
    features : dict[str, array-like] or DataFrame
        Features table where each row corresponds to a point and each column
        is a feature.
    feature_defaults : dict[str, Any] or DataFrame
        The default value of each feature in a table with one row.
    properties : dict {str: array (N,)}, DataFrame
        Properties for each point. Each property should be an array of length N,
        where N is the number of points.
    property_choices : dict {str: array (N,)}
        possible values for each property.
    text : str, dict
        Text to be displayed with the points. If text is set to a key in properties,
        the value of that property will be displayed. Multiple properties can be
        composed using f-string-like syntax (e.g., '{property_1}, {float_property:.2f}).
        A dictionary can be provided with keyword arguments to set the text values
        and display properties. See TextManager.__init__() for the valid keyword arguments.
        For example usage, see /napari/examples/add_points_with_text.py.
    symbol : str, array
        Symbols to be used for the point markers. Must be one of the
        following: arrow, clobber, cross, diamond, disc, hbar, ring,
        square, star, tailed_arrow, triangle_down, triangle_up, vbar, x.
    size : float, array
        Size of the point marker in data pixels. If given as a scalar, all points are made
        the same size. If given as an array, size must be the same or broadcastable
        to the same shape as the data.
    border_width : float, array
        Width of the symbol border in pixels.
    border_width_is_relative : bool
        If enabled, border_width is interpreted as a fraction of the point size.
    border_color : str, array-like, dict
        Color of the point marker border. Numeric color values should be RGB(A).
    border_color_cycle : np.ndarray, list
        Cycle of colors (provided as string name, RGB, or RGBA) to map to border_color if a
        categorical attribute is used color the vectors.
    border_colormap : str, napari.utils.Colormap
        Colormap to set border_color if a continuous attribute is used to set face_color.
    border_contrast_limits : None, (float, float)
        clims for mapping the property to a color map. These are the min and max value
        of the specified property that are mapped to 0 and 1, respectively.
        The default value is None. If set the none, the clims will be set to
        (property.min(), property.max())
    face_color : str, array-like, dict
        Color of the point marker body. Numeric color values should be RGB(A).
    face_color_cycle : np.ndarray, list
        Cycle of colors (provided as string name, RGB, or RGBA) to map to face_color if a
        categorical attribute is used color the vectors.
    face_colormap : str, napari.utils.Colormap
        Colormap to set face_color if a continuous attribute is used to set face_color.
    face_contrast_limits : None, (float, float)
        clims for mapping the property to a color map. These are the min and max value
        of the specified property that are mapped to 0 and 1, respectively.
        The default value is None. If set the none, the clims will be set to
        (property.min(), property.max())
    out_of_slice_display : bool
        If True, renders points not just in central plane but also slightly out of slice
        according to specified point marker size.
    n_dimensional : bool
        This property will soon be deprecated in favor of 'out_of_slice_display'.
        Use that instead.
    name : str
        Name of the layer.
    metadata : dict
        Layer metadata.
    scale : tuple of float
        Scale factors for the layer.
    translate : tuple of float
        Translation values for the layer.
    rotate : float, 3-tuple of float, or n-D array.
        If a float convert into a 2D rotation matrix using that value as an
        angle. If 3-tuple convert into a 3D rotation matrix, using a yaw,
        pitch, roll convention. Otherwise assume an nD rotation. Angles are
        assumed to be in degrees. They can be converted from radians with
        np.degrees if needed.
    shear : 1-D array or n-D array
        Either a vector of upper triangular values, or an nD shear matrix with
        ones along the main diagonal.
    affine : n-D array or napari.utils.transforms.Affine
        (N+1, N+1) affine transformation matrix in homogeneous coordinates.
        The first (N, N) entries correspond to a linear transform and
        the final column is a length N translation vector and a 1 or a napari
        `Affine` transform object. Applied as an extra transform on top of the
        provided scale, rotate, and shear values.
    opacity : float
        Opacity of the layer visual, between 0.0 and 1.0.
    blending : str
        One of a list of preset blending modes that determines how RGB and
        alpha values of the layer visual get mixed. Allowed values are
        {'opaque', 'translucent', and 'additive'}.
    visible : bool
        Whether the layer visual is currently being displayed.
    cache : bool
        Whether slices of out-of-core datasets should be cached upon retrieval.
        Currently, this only applies to dask arrays.
    shading : str, Shading
        Render lighting and shading on points. Options are:

        * 'none'
          No shading is added to the points.
        * 'spherical'
          Shading and depth buffer are changed to give a 3D spherical look to the points
    antialiasing: float
        Amount of antialiasing in canvas pixels.
    canvas_size_limits : tuple of float
        Lower and upper limits for the size of points in canvas pixels.
    shown : 1-D array of bool
        Whether to show each point.

    Attributes
    ----------
    data : array (N, D)
        Coordinates for N points in D dimensions.
    features : DataFrame-like
        Features table where each row corresponds to a point and each column
        is a feature.
    feature_defaults : DataFrame-like
        Stores the default value of each feature in a table with one row.
    properties : dict {str: array (N,)} or DataFrame
        Annotations for each point. Each property should be an array of length N,
        where N is the number of points.
    text : str
        Text to be displayed with the points. If text is set to a key in properties, the value of
        that property will be displayed. Multiple properties can be composed using f-string-like
        syntax (e.g., '{property_1}, {float_property:.2f}).
        For example usage, see /napari/examples/add_points_with_text.py.
    symbol : array of str
        Array of symbols for each point.
    size : array (N, D)
        Array of sizes for each point in each dimension. Must have the same
        shape as the layer `data`.
    border_width : array (N,)
        Width of the marker borders in pixels for all points
    border_width : array (N,)
        Width of the marker borders for all points as a fraction of their size.
    border_color : Nx4 numpy array
        Array of border color RGBA values, one for each point.
    border_color_cycle : np.ndarray, list
        Cycle of colors (provided as string name, RGB, or RGBA) to map to border_color if a
        categorical attribute is used color the vectors.
    border_colormap : str, napari.utils.Colormap
        Colormap to set border_color if a continuous attribute is used to set face_color.
    border_contrast_limits : None, (float, float)
        clims for mapping the property to a color map. These are the min and max value
        of the specified property that are mapped to 0 and 1, respectively.
        The default value is None. If set the none, the clims will be set to
        (property.min(), property.max())
    face_color : Nx4 numpy array
        Array of face color RGBA values, one for each point.
    face_color_cycle : np.ndarray, list
        Cycle of colors (provided as string name, RGB, or RGBA) to map to face_color if a
        categorical attribute is used color the vectors.
    face_colormap : str, napari.utils.Colormap
        Colormap to set face_color if a continuous attribute is used to set face_color.
    face_contrast_limits : None, (float, float)
        clims for mapping the property to a color map. These are the min and max value
        of the specified property that are mapped to 0 and 1, respectively.
        The default value is None. If set the none, the clims will be set to
        (property.min(), property.max())
    current_symbol : Symbol
        Symbol for the next point to be added or the currently selected points.
    current_size : float
        Size of the marker for the next point to be added or the currently
        selected point.
    current_border_width : float
        border width of the marker for the next point to be added or the currently
        selected point.
    current_border_color : str
        border color of the marker border for the next point to be added or the currently
        selected point.
    current_face_color : str
        Face color of the marker border for the next point to be added or the currently
        selected point.
    out_of_slice_display : bool
        If True, renders points not just in central plane but also slightly out of slice
        according to specified point marker size.
    selected_data : Selection
        Integer indices of any selected points.
    mode : str
        Interactive mode. The normal, default mode is PAN_ZOOM, which
        allows for normal interactivity with the canvas.

        In ADD mode clicks of the cursor add points at the clicked location.

        In SELECT mode the cursor can select points by clicking on them or
        by dragging a box around them. Once selected points can be moved,
        have their properties edited, or be deleted.
    face_color_mode : str
        Face color setting mode.

        DIRECT (default mode) allows each point to be set arbitrarily

        CYCLE allows the color to be set via a color cycle over an attribute

        COLORMAP allows color to be set via a color map over an attribute
    border_color_mode : str
        border color setting mode.

        DIRECT (default mode) allows each point to be set arbitrarily

        CYCLE allows the color to be set via a color cycle over an attribute

        COLORMAP allows color to be set via a color map over an attribute
    shading : Shading
        Shading mode.
    antialiasing: float
        Amount of antialiasing in canvas pixels.
    canvas_size_limits : tuple of float
        Lower and upper limits for the size of points in canvas pixels.
    shown : 1-D array of bool
        Whether each point is shown.

    Notes
    -----
    _view_data : array (M, D)
        coordinates of points in the currently viewed slice.
    _view_size : array (M, )
        Size of the point markers in the currently viewed slice.
    _view_symbol : array (M, )
        Symbols of the point markers in the currently viewed slice.
    _view_border_width : array (M, )
        border width of the point markers in the currently viewed slice.
    _indices_view : array (M, )
        Integer indices of the points in the currently viewed slice and are shown.
    _selected_view :
        Integer indices of selected points in the currently viewed slice within
        the `_view_data` array.
    _selected_box : array (4, 2) or None
        Four corners of any box either around currently selected points or
        being created during a drag action. Starting in the top left and
        going clockwise.
    _drag_start : list or None
        Coordinates of first cursor click during a drag action. Gets reset to
        None after dragging is done.
    """

    _modeclass = Mode

    _drag_modes = {
        Mode.PAN_ZOOM: no_op,
        Mode.TRANSFORM: transform_with_box,
        Mode.ADD: add,
        Mode.SELECT: select,
    }

    _move_modes = {
        Mode.PAN_ZOOM: no_op,
        Mode.TRANSFORM: highlight_box_handles,
        Mode.ADD: no_op,
        Mode.SELECT: highlight,
    }
    _cursor_modes = {
        Mode.PAN_ZOOM: 'standard',
        Mode.TRANSFORM: 'standard',
        Mode.ADD: 'crosshair',
        Mode.SELECT: 'standard',
    }

    # TODO  write better documentation for border_color and face_color

    # The max number of points that will ever be used to render the thumbnail
    # If more points are present then they are randomly subsampled
    _max_points_thumbnail = 1024

    def __init__(
        self,
        data=None,
        *,
        ndim=None,
        features=None,
        feature_defaults=None,
        properties=None,
        text=None,
        symbol='o',
        size=10,
        border_width=0.05,
        border_width_is_relative=True,
        border_color='dimgray',
        border_color_cycle=None,
        border_colormap='viridis',
        border_contrast_limits=None,
        face_color='white',
        face_color_cycle=None,
        face_colormap='viridis',
        face_contrast_limits=None,
        out_of_slice_display=False,
        n_dimensional=None,
        name=None,
        metadata=None,
        scale=None,
        translate=None,
        rotate=None,
        shear=None,
        affine=None,
        opacity=1,
        blending='translucent',
        visible=True,
        cache=True,
        property_choices=None,
        experimental_clipping_planes=None,
        shading='none',
        canvas_size_limits=(2, 10000),
        antialiasing=1,
        shown=True,
    ) -> None:
        # Indices of selected points
        self._selected_data_stored = set()
        self._selected_data_history = set()
        # Indices of selected points within the currently viewed slice
        self._selected_view = []
        # Index of hovered point
        self._value = None
        self._value_stored = None
        self._highlight_index = []
        self._highlight_box = None

        self._drag_start = None
        self._drag_normal = None
        self._drag_up = None

        # initialize view data
        self.__indices_view = np.empty(0, int)
        self._view_size_scale = []

        self._drag_box = None
        self._drag_box_stored = None
        self._is_selecting = False
        self._clipboard = {}
        self._round_index = False

        super().__init__(
            data,
            ndim,
            name=name,
            metadata=metadata,
            scale=scale,
            translate=translate,
            rotate=rotate,
            shear=shear,
            affine=affine,
            opacity=opacity,
            blending=blending,
            visible=visible,
            cache=cache,
            experimental_clipping_planes=experimental_clipping_planes,
        )

        self.events.add(
            size=Event,
            current_size=Event,
            border_width=Event,
            current_border_width=Event,
            border_width_is_relative=Event,
            face_color=Event,
            current_face_color=Event,
            border_color=Event,
            current_border_color=Event,
            properties=Event,
            current_properties=Event,
            symbol=Event,
            current_symbol=Event,
            out_of_slice_display=Event,
            n_dimensional=Event,
            highlight=Event,
            shading=Event,
            antialiasing=Event,
            canvas_size_limits=Event,
            features=Event,
            feature_defaults=Event,
        )

        self._feature_table = _FeatureTable.from_layer(
            features=features,
            feature_defaults=feature_defaults,
            properties=properties,
            property_choices=property_choices,
            num_data=len(self._points_data),
        )

        self._text = TextManager._from_layer(
            text=text,
            features=self.features,
        )

        self._border_width_is_relative = False
        self._shown = np.empty(0).astype(bool)

        # Indices of selected points
        self._selected_data: Selection[int] = Selection()
        self._selected_data_stored = set()
        self._selected_data_history = set()
        # Indices of selected points within the currently viewed slice
        self._selected_view = []

        # The following point properties are for the new points that will
        # be added. For any given property, if a list is passed to the
        # constructor so each point gets its own value then the default
        # value is used when adding new points
        self._current_size = np.asarray(size) if np.isscalar(size) else 10
        self._current_border_width = (
            np.asarray(border_width) if np.isscalar(border_width) else 0.1
        )
        self.current_symbol = (
            np.asarray(symbol) if np.isscalar(symbol) else 'o'
        )

        # Index of hovered point
        self._value = None
        self._value_stored = None
        self._mode = Mode.PAN_ZOOM
        self._status = self.mode

        color_properties = (
            self._feature_table.properties()
            if len(self._points_data)
            else self._feature_table.currents()
        )
        self._border = ColorManager._from_layer_kwargs(
            n_colors=len(data),
            colors=border_color,
            continuous_colormap=border_colormap,
            contrast_limits=border_contrast_limits,
            categorical_colormap=border_color_cycle,
            properties=color_properties,
        )
        self._face = ColorManager._from_layer_kwargs(
            n_colors=len(data),
            colors=face_color,
            continuous_colormap=face_colormap,
            contrast_limits=face_contrast_limits,
            categorical_colormap=face_color_cycle,
            properties=color_properties,
        )

        if n_dimensional is not None:
            self._out_of_slice_display = n_dimensional
        else:
            self._out_of_slice_display = out_of_slice_display

        # Save the point style params
        self.size = size
        self.shown = shown
        self.symbol = symbol
        self.border_width = border_width
        self.border_width_is_relative = border_width_is_relative

        self.canvas_size_limits = canvas_size_limits
        self.shading = shading
        self.antialiasing = antialiasing

        # Trigger generation of view slice and thumbnail
        self.refresh()

    @property
    def _points_data(self) -> np.ndarray:
        """Spatialy distributed coordinates."""
        raise NotImplementedError

    @property
    def data(self) -> Any:
        raise NotImplementedError

    @data.setter
    def data(self, data: Any) -> None:
        raise NotImplementedError

    def _on_selection(self, selected):
        if selected:
            self._set_highlight()
        else:
            self._highlight_box = None
            self._highlight_index = []
            self.events.highlight()

    @property
    def features(self):
        """Dataframe-like features table.

        It is an implementation detail that this is a `pandas.DataFrame`. In the future,
        we will target the currently-in-development Data API dataframe protocol [1].
        This will enable us to use alternate libraries such as xarray or cuDF for
        additional features without breaking existing usage of this.

        If you need to specifically rely on the pandas API, please coerce this to a
        `pandas.DataFrame` using `features_to_pandas_dataframe`.

        References
        ----------
        .. [1]: https://data-apis.org/dataframe-protocol/latest/API.html
        """
        return self._feature_table.values

    @features.setter
    def features(
        self,
        features: Union[Dict[str, np.ndarray], pd.DataFrame],
    ) -> None:
        self._feature_table.set_values(
            features, num_data=len(self._points_data)
        )
        self._update_color_manager(
            self._face, self._feature_table, "face_color"
        )
        self._update_color_manager(
            self._border, self._feature_table, "border_color"
        )
        self.text.refresh(self.features)
        self.events.properties()
        self.events.features()

    @property
    def feature_defaults(self):
        """Dataframe-like with one row of feature default values.

        See `features` for more details on the type of this property.
        """
        return self._feature_table.defaults

    @feature_defaults.setter
    def feature_defaults(
        self, defaults: Union[Dict[str, Any], pd.DataFrame]
    ) -> None:
        self._feature_table.set_defaults(defaults)
        current_properties = self.current_properties
        self._border._update_current_properties(current_properties)
        self._face._update_current_properties(current_properties)
        self.events.current_properties()
        self.events.feature_defaults()

    @property
    def property_choices(self) -> Dict[str, np.ndarray]:
        return self._feature_table.choices()

    @property
    def properties(self) -> Dict[str, np.ndarray]:
        """dict {str: np.ndarray (N,)}, DataFrame: Annotations for each point"""
        return self._feature_table.properties()

    @staticmethod
    def _update_color_manager(color_manager, feature_table, name):
        if color_manager.color_properties is not None:
            color_name = color_manager.color_properties.name
            if color_name not in feature_table.values:
                color_manager.color_mode = ColorMode.DIRECT
                color_manager.color_properties = None
                warnings.warn(
                    trans._(
                        'property used for {name} dropped',
                        deferred=True,
                        name=name,
                    ),
                    RuntimeWarning,
                )
            else:
                color_manager.color_properties = {
                    'name': color_name,
                    'values': feature_table.values[color_name].to_numpy(),
                    'current_value': feature_table.defaults[color_name][0],
                }

    @properties.setter
    def properties(
        self, properties: Union[Dict[str, Array], pd.DataFrame, None]
    ):
        self.features = properties

    @property
    def current_properties(self) -> Dict[str, np.ndarray]:
        """dict{str: np.ndarray(1,)}: properties for the next added point."""
        return self._feature_table.currents()

    @current_properties.setter
    def current_properties(self, current_properties):
        update_indices = None
        if self._update_properties and len(self.selected_data) > 0:
            update_indices = list(self.selected_data)
        self._feature_table.set_currents(
            current_properties, update_indices=update_indices
        )
        current_properties = self.current_properties
        self._border._update_current_properties(current_properties)
        self._face._update_current_properties(current_properties)
        self.events.current_properties()
        self.events.feature_defaults()
        if update_indices is not None:
            self.events.properties()
            self.events.features()

    @property
    def text(self) -> TextManager:
        """TextManager: the TextManager object containing containing the text properties"""
        return self._text

    @text.setter
    def text(self, text):
        self._text._update_from_layer(
            text=text,
            features=self.features,
        )

    def refresh_text(self):
        """Refresh the text values.

        This is generally used if the features were updated without changing the data
        """
        self.text.refresh(self.features)

    @abstractmethod
    def _get_ndim(self) -> int:
        """Determine number of dimensions of the layer."""
        raise NotImplementedError

    @property
    def _extent_data(self) -> np.ndarray:
        """Extent of layer in data coordinates.

        Returns
        -------
        extent_data : array, shape (2, D)
        """
        if len(self._points_data) == 0:
            extrema = np.full((2, self.ndim), np.nan)
        else:
            maxs = np.max(self._points_data, axis=0)
            mins = np.min(self._points_data, axis=0)
            extrema = np.vstack([mins, maxs])
        return extrema.astype(float)

    @property
    def _extent_data_augmented(self):
        # _extent_data is a property that returns a new/copied array, which
        # is safe to modify below
        extent = self._extent_data
        if len(self.size) == 0:
            return extent

        max_point_size = np.max(self.size)
        extent[0] -= max_point_size / 2
        extent[1] += max_point_size / 2
        return extent

    @property
    def out_of_slice_display(self) -> bool:
        """bool: renders points slightly out of slice."""
        return self._out_of_slice_display

    @out_of_slice_display.setter
    def out_of_slice_display(self, out_of_slice_display: bool) -> None:
        self._out_of_slice_display = bool(out_of_slice_display)
        self.events.out_of_slice_display()
        self.events.n_dimensional()
        self.refresh()

    @property
    def n_dimensional(self) -> bool:
        """
        This property will soon be deprecated in favor of `out_of_slice_display`. Use that instead.
        """
        return self._out_of_slice_display

    @n_dimensional.setter
    def n_dimensional(self, value: bool) -> None:
        self.out_of_slice_display = value

    @property
    def symbol(self) -> np.ndarray:
        """str: symbol used for all point markers."""
        return self._symbol

    @symbol.setter
    def symbol(self, symbol: Union[str, np.ndarray, list]) -> None:
        symbol = np.broadcast_to(symbol, len(self._points_data))
        self._symbol = coerce_symbols(symbol)
        self.events.symbol()
        self.events.highlight()

    @property
    def current_symbol(self) -> Union[int, float]:
        """float: symbol of marker for the next added point."""
        return self._current_symbol

    @current_symbol.setter
    def current_symbol(self, symbol: Union[None, float]) -> None:
        symbol = coerce_symbols(np.array([symbol]))[0]
        self._current_symbol = symbol
        if self._update_properties and len(self.selected_data) > 0:
            self.symbol[list(self.selected_data)] = symbol
            self.events.symbol()
        self.events.current_symbol()

    @property
    def size(self) -> np.ndarray:
        """(N, D) array: size of all N points in D dimensions."""
        return self._size

    @size.setter
    def size(self, size: Union[int, float, np.ndarray, list]) -> None:
        try:
            self._size = np.broadcast_to(size, self._points_data.shape).copy()
        except ValueError as e:
            try:
                self._size = np.broadcast_to(
                    size, self._points_data.shape[::-1]
                ).T.copy()
            except ValueError:
                raise ValueError(
                    trans._(
                        "Size of shape {size_shape} is not compatible for broadcasting "
                        "with shape {points_shape}",
                        size_shape=size.shape,
                        points_shape=self._points_data.shape,
                        deferred=True,
                    )
                ) from e
        self._clear_extent_augmented()
        self.refresh()

    @property
    def current_size(self) -> Union[int, float]:
        """float: size of marker for the next added point."""
        return self._current_size

    @current_size.setter
    def current_size(self, size: Union[None, float]) -> None:
        if (isinstance(size, numbers.Number) and size < 0) or (
            isinstance(size, list) and min(size) < 0
        ):
            warnings.warn(
                message=trans._(
                    'current_size value must be positive, value will be left at {value}.',
                    deferred=True,
                    value=self.current_size,
                ),
                category=RuntimeWarning,
            )
            size = self.current_size
        self._current_size = size
        if self._update_properties and len(self.selected_data) > 0:
            for i in self.selected_data:
                self.size[i, :] = (self.size[i, :] > 0) * size
            self._clear_extent_augmented()
            self.refresh()
            self.events.size()
        self.events.current_size()

    @property
    def antialiasing(self) -> float:
        """Amount of antialiasing in canvas pixels."""
        return self._antialiasing

    @antialiasing.setter
    def antialiasing(self, value: float):
        """Set the amount of antialiasing in canvas pixels.

        Values can only be positive.
        """
        if value < 0:
            warnings.warn(
                message=trans._(
                    'antialiasing value must be positive, value will be set to 0.',
                    deferred=True,
                ),
                category=RuntimeWarning,
            )
        self._antialiasing = max(0, value)
        self.events.antialiasing(value=self._antialiasing)

    @property
    def shading(self) -> Shading:
        """shading mode."""
        return self._shading

    @shading.setter
    def shading(self, value):
        self._shading = Shading(value)
        self.events.shading()

    @property
    def canvas_size_limits(self) -> Tuple[float, float]:
        """Limit the canvas size of points"""
        return self._canvas_size_limits

    @canvas_size_limits.setter
    def canvas_size_limits(self, value):
        self._canvas_size_limits = float(value[0]), float(value[1])
        self.events.canvas_size_limits()

    @property
    def shown(self):
        """
        Boolean array determining which points to show
        """
        return self._shown

    @shown.setter
    def shown(self, shown):
        self._shown = np.broadcast_to(shown, len(self._points_data)).astype(
            bool
        )
        self.refresh()

    @property
    def border_width(self) -> np.ndarray:
        """(N, D) array: border_width of all N points."""
        return self._border_width

    @border_width.setter
    def border_width(
        self, border_width: Union[int, float, np.ndarray, list]
    ) -> None:
        # broadcast to np.array
        border_width = np.broadcast_to(
            border_width, len(self._points_data)
        ).copy()

        # border width cannot be negative
        if np.any(border_width < 0):
            raise ValueError(
                trans._(
                    'All border_width must be > 0',
                    deferred=True,
                )
            )
        # if relative border width is enabled, border_width must be between 0 and 1
        if self.border_width_is_relative and np.any(border_width > 1):
            raise ValueError(
                trans._(
                    'All border_width must be between 0 and 1 if border_width_is_relative is enabled',
                    deferred=True,
                )
            )

        self._border_width: np.ndarray = border_width
        self.refresh()

    @property
    def border_width_is_relative(self) -> bool:
        """bool: treat border_width as a fraction of point size."""
        return self._border_width_is_relative

    @border_width_is_relative.setter
    def border_width_is_relative(self, border_width_is_relative: bool) -> None:
        if border_width_is_relative and np.any(
            (self.border_width > 1) | (self.border_width < 0)
        ):
            raise ValueError(
                trans._(
                    'border_width_is_relative can only be enabled if border_width is between 0 and 1',
                    deferred=True,
                )
            )
        self._border_width_is_relative = border_width_is_relative
        self.events.border_width_is_relative()

    @property
    def current_border_width(self) -> Union[int, float]:
        """float: border_width of marker for the next added point."""
        return self._current_border_width

    @current_border_width.setter
    def current_border_width(self, border_width: Union[None, float]) -> None:
        self._current_border_width = border_width
        if self._update_properties and len(self.selected_data) > 0:
            for i in self.selected_data:
                self.border_width[i] = (
                    self.border_width[i] > 0
                ) * border_width
            self.refresh()
            self.events.border_width()
        self.events.current_border_width()

    @property
    def border_color(self) -> np.ndarray:
        """(N x 4) np.ndarray: Array of RGBA border colors for each point"""
        return self._border.colors

    @border_color.setter
    def border_color(self, border_color):
        self._border._set_color(
            color=border_color,
            n_colors=len(self._points_data),
            properties=self.properties,
            current_properties=self.current_properties,
        )
        self.events.border_color()

    @property
    def border_color_cycle(self) -> np.ndarray:
        """Union[list, np.ndarray] :  Color cycle for border_color.
        Can be a list of colors defined by name, RGB or RGBA
        """
        return self._border.categorical_colormap.fallback_color.values

    @border_color_cycle.setter
    def border_color_cycle(self, border_color_cycle: Union[list, np.ndarray]):
        self._border.categorical_colormap = border_color_cycle

    @property
    def border_colormap(self) -> Colormap:
        """Return the colormap to be applied to a property to get the border color.

        Returns
        -------
        colormap : napari.utils.Colormap
            The Colormap object.
        """
        return self._border.continuous_colormap

    @border_colormap.setter
    def border_colormap(self, colormap: ValidColormapArg):
        self._border.continuous_colormap = colormap

    @property
    def border_contrast_limits(self) -> Tuple[float, float]:
        """None, (float, float): contrast limits for mapping
        the border_color colormap property to 0 and 1
        """
        return self._border.contrast_limits

    @border_contrast_limits.setter
    def border_contrast_limits(
        self, contrast_limits: Union[None, Tuple[float, float]]
    ):
        self._border.contrast_limits = contrast_limits

    @property
    def current_border_color(self) -> str:
        """str: border color of marker for the next added point or the selected point(s)."""
        hex_ = rgb_to_hex(self._border.current_color)[0]
        return hex_to_name.get(hex_, hex_)

    @current_border_color.setter
    def current_border_color(self, border_color: ColorType) -> None:
        if self._update_properties and len(self.selected_data) > 0:
            update_indices = list(self.selected_data)
        else:
            update_indices = []
        self._border._update_current_color(
            border_color, update_indices=update_indices
        )
        self.events.current_border_color()

    @property
    def border_color_mode(self) -> str:
        """str: border color setting mode

        DIRECT (default mode) allows each point to be set arbitrarily

        CYCLE allows the color to be set via a color cycle over an attribute

        COLORMAP allows color to be set via a color map over an attribute
        """
        return self._border.color_mode

    @border_color_mode.setter
    def border_color_mode(self, border_color_mode: Union[str, ColorMode]):
        self._set_color_mode(border_color_mode, 'border')

    @property
    def face_color(self) -> np.ndarray:
        """(N x 4) np.ndarray: Array of RGBA face colors for each point"""
        return self._face.colors

    @face_color.setter
    def face_color(self, face_color):
        self._face._set_color(
            color=face_color,
            n_colors=len(self._points_data),
            properties=self.properties,
            current_properties=self.current_properties,
        )
        self.events.face_color()

    @property
    def face_color_cycle(self) -> np.ndarray:
        """Union[np.ndarray, cycle]:  Color cycle for face_color
        Can be a list of colors defined by name, RGB or RGBA
        """
        return self._face.categorical_colormap.fallback_color.values

    @face_color_cycle.setter
    def face_color_cycle(self, face_color_cycle: Union[np.ndarray, cycle]):
        self._face.categorical_colormap = face_color_cycle

    @property
    def face_colormap(self) -> Colormap:
        """Return the colormap to be applied to a property to get the face color.

        Returns
        -------
        colormap : napari.utils.Colormap
            The Colormap object.
        """
        return self._face.continuous_colormap

    @face_colormap.setter
    def face_colormap(self, colormap: ValidColormapArg):
        self._face.continuous_colormap = colormap

    @property
    def face_contrast_limits(self) -> Union[None, Tuple[float, float]]:
        """None, (float, float) : clims for mapping the face_color
        colormap property to 0 and 1
        """
        return self._face.contrast_limits

    @face_contrast_limits.setter
    def face_contrast_limits(
        self, contrast_limits: Union[None, Tuple[float, float]]
    ):
        self._face.contrast_limits = contrast_limits

    @property
    def current_face_color(self) -> str:
        """Face color of marker for the next added point or the selected point(s)."""
        hex_ = rgb_to_hex(self._face.current_color)[0]
        return hex_to_name.get(hex_, hex_)

    @current_face_color.setter
    def current_face_color(self, face_color: ColorType) -> None:
        if self._update_properties and len(self.selected_data) > 0:
            update_indices = list(self.selected_data)
        else:
            update_indices = []
        self._face._update_current_color(
            face_color, update_indices=update_indices
        )
        self.events.current_face_color()

    @property
    def face_color_mode(self) -> str:
        """str: Face color setting mode

        DIRECT (default mode) allows each point to be set arbitrarily

        CYCLE allows the color to be set via a color cycle over an attribute

        COLORMAP allows color to be set via a color map over an attribute
        """
        return self._face.color_mode

    @face_color_mode.setter
    def face_color_mode(self, face_color_mode):
        self._set_color_mode(face_color_mode, 'face')

    def _set_color_mode(
        self, color_mode: Union[ColorMode, str], attribute: str
    ):
        """Set the face_color_mode or border_color_mode property

        Parameters
        ----------
        color_mode : str, ColorMode
            The value for setting border or face_color_mode. If color_mode is a string,
            it should be one of: 'direct', 'cycle', or 'colormap'
        attribute : str in {'border', 'face'}
            The name of the attribute to set the color of.
            Should be 'border' for border_color_mode or 'face' for face_color_mode.
        """
        color_mode = ColorMode(color_mode)
        color_manager = getattr(self, f'_{attribute}')

        if color_mode == ColorMode.DIRECT:
            color_manager.color_mode = color_mode
        elif color_mode in (ColorMode.CYCLE, ColorMode.COLORMAP):
            if color_manager.color_properties is not None:
                color_property = color_manager.color_properties.name
            else:
                color_property = ''
            if color_property == '':
                if self.features.shape[1] > 0:
                    new_color_property = next(iter(self.features))
                    color_manager.color_properties = {
                        'name': new_color_property,
                        'values': self.features[new_color_property].to_numpy(),
                        'current_value': np.squeeze(
                            self.current_properties[new_color_property]
                        ),
                    }
                    warnings.warn(
                        trans._(
                            '_{attribute}_color_property was not set, setting to: {new_color_property}',
                            deferred=True,
                            attribute=attribute,
                            new_color_property=new_color_property,
                        )
                    )
                else:
                    raise ValueError(
                        trans._(
                            'There must be a valid Points.properties to use {color_mode}',
                            deferred=True,
                            color_mode=color_mode,
                        )
                    )

            # ColorMode.COLORMAP can only be applied to numeric properties
            color_property = color_manager.color_properties.name
            if (color_mode == ColorMode.COLORMAP) and not issubclass(
                self.features[color_property].dtype.type, np.number
            ):
                raise TypeError(
                    trans._(
                        'selected property must be numeric to use ColorMode.COLORMAP',
                        deferred=True,
                    )
                )
            color_manager.color_mode = color_mode

    def refresh_colors(self, update_color_mapping: bool = False):
        """Calculate and update face and border colors if using a cycle or color map

        Parameters
        ----------
        update_color_mapping : bool
            If set to True, the function will recalculate the color cycle map
            or colormap (whichever is being used). If set to False, the function
            will use the current color cycle map or color map. For example, if you
            are adding/modifying points and want them to be colored with the same
            mapping as the other points (i.e., the new points shouldn't affect
            the color cycle map or colormap), set ``update_color_mapping=False``.
            Default value is False.
        """
        self._border._refresh_colors(self.properties, update_color_mapping)
        self._face._refresh_colors(self.properties, update_color_mapping)

    def _get_state(self):
        """Get dictionary of layer state.

        Returns
        -------
        state : dict
            Dictionary of layer state.
        """

        # must be self.data and not self._points_data
        # self._points_data includes invalid nodes from graph buffer.
        not_empty = len(self.data) > 0
        state = self._get_base_state()
        state.update(
            {
                'symbol': self.symbol if not_empty else [self.current_symbol],
                'border_width': self.border_width,
                'border_width_is_relative': self.border_width_is_relative,
                'face_color': self.face_color
                if not_empty
                else [self.current_face_color],
                'face_color_cycle': self.face_color_cycle,
                'face_colormap': self.face_colormap.name,
                'face_contrast_limits': self.face_contrast_limits,
                'border_color': self.border_color
                if not_empty
                else [self.current_border_color],
                'border_color_cycle': self.border_color_cycle,
                'border_colormap': self.border_colormap.name,
                'border_contrast_limits': self.border_contrast_limits,
                'properties': self.properties,
                'property_choices': self.property_choices,
                'text': self.text.dict(),
                'out_of_slice_display': self.out_of_slice_display,
                'n_dimensional': self.out_of_slice_display,
                'size': self.size,
                'ndim': self.ndim,
                'data': self.data,
                'features': self.features,
                'feature_defaults': self.feature_defaults,
                'shading': self.shading,
                'antialiasing': self.antialiasing,
                'canvas_size_limits': self.canvas_size_limits,
                'shown': self.shown,
            }
        )
        return state

    @property
    def selected_data(self) -> Selection[int]:
        """set: set of currently selected points."""
        return self._selected_data

    @selected_data.setter
    def selected_data(self, selected_data: Sequence[int]) -> None:
        self._selected_data.clear()
        self._selected_data.update(set(selected_data))
        self._selected_view = list(
            np.intersect1d(
                np.array(list(self._selected_data)),
                self._indices_view,
                return_indices=True,
            )[2]
        )

        # Update properties based on selected points
        if not len(self._selected_data):
            self._set_highlight()
            return
        index = list(self._selected_data)
        if (
            unique_border_color := _unique_element(self.border_color[index])
        ) is not None:
            with self.block_update_properties():
                self.current_border_color = unique_border_color

        if (
            unique_face_color := _unique_element(self.face_color[index])
        ) is not None:
            with self.block_update_properties():
                self.current_face_color = unique_face_color

        # Calculate the mean size across the displayed dimensions for
        # each point to be consistent with `_view_size`.
        mean_size = np.mean(
            self.size[np.ix_(index, self._slice_input.displayed)], axis=1
        )
        if (unique_size := _unique_element(mean_size)) is not None:
            with self.block_update_properties():
                self.current_size = unique_size

        if (
            unique_border_width := _unique_element(self.border_width[index])
        ) is not None:
            with self.block_update_properties():
                self.current_border_width = unique_border_width
        if (unique_symbol := _unique_element(self.symbol[index])) is not None:
            with self.block_update_properties():
                self.current_symbol = unique_symbol

        unique_properties = {}
        for k, v in self.properties.items():
            unique_properties[k] = _unique_element(v[index])

        if all(p is not None for p in unique_properties.values()):
            with self.block_update_properties():
                self.current_properties = unique_properties
        self._set_highlight()

    def interaction_box(self, index) -> Optional[np.ndarray]:
        """Create the interaction box around a list of points in view.

        Parameters
        ----------
        index : list
            List of points around which to construct the interaction box.

        Returns
        -------
        box : np.ndarray or None
            4x2 array of corners of the interaction box in clockwise order
            starting in the upper-left corner.
        """
        if len(index) > 0:
            data = self._view_data[index]
            size = self._view_size[index]
            data = points_to_squares(data, size)
            return create_box(data)
        return None

    @Layer.mode.getter
    def mode(self) -> str:
        """str: Interactive mode

        Interactive mode. The normal, default mode is PAN_ZOOM, which
        allows for normal interactivity with the canvas.

        In ADD mode clicks of the cursor add points at the clicked location.

        In SELECT mode the cursor can select points by clicking on them or
        by dragging a box around them. Once selected points can be moved,
        have their properties edited, or be deleted.
        """
        return str(self._mode)

    def _mode_setter_helper(self, mode):
        mode = super()._mode_setter_helper(mode)
        if mode == self._mode:
            return mode

        if mode == Mode.ADD:
            self.selected_data = set()
            self.mouse_pan = True
        elif mode != Mode.SELECT or self._mode != Mode.SELECT:
            self._selected_data_stored = set()

        self._set_highlight()
        return mode

    @property
    def _indices_view(self):
        return self.__indices_view

    @_indices_view.setter
    def _indices_view(self, value):
        if len(self._shown) == 0:
            self.__indices_view = np.empty(0, int)
        else:
            self.__indices_view = value[self.shown[value]]

    @property
    def _view_data(self) -> np.ndarray:
        """Get the coords of the points in view

        Returns
        -------
        view_data : (N x D) np.ndarray
            Array of coordinates for the N points in view
        """
        if len(self._indices_view) > 0:
            data = self._points_data[
                np.ix_(self._indices_view, self._slice_input.displayed)
            ]
        else:
            # if no points in this slice send dummy data
            data = np.zeros((0, self._slice_input.ndisplay))

        return data

    @property
    def _view_text(self) -> np.ndarray:
        """Get the values of the text elements in view

        Returns
        -------
        text : (N x 1) np.ndarray
            Array of text strings for the N text elements in view
        """
        # This may be triggered when the string encoding instance changed,
        # in which case it has no cached values, so generate them here.
        self.text.string._apply(self.features)
        return self.text.view_text(self._indices_view)

    @property
    def _view_text_coords(self) -> Tuple[np.ndarray, str, str]:
        """Get the coordinates of the text elements in view

        Returns
        -------
        text_coords : (N x D) np.ndarray
            Array of coordinates for the N text elements in view
        anchor_x : str
            The vispy text anchor for the x axis
        anchor_y : str
            The vispy text anchor for the y axis
        """
        return self.text.compute_text_coords(
            self._view_data,
            self._slice_input.ndisplay,
            self._slice_input.order,
        )

    @property
    def _view_text_color(self) -> np.ndarray:
        """Get the colors of the text elements at the given indices."""
        self.text.color._apply(self.features)
        return self.text._view_color(self._indices_view)

    @property
    def _view_size(self) -> np.ndarray:
        """Get the sizes of the points in view

        Returns
        -------
        view_size : (N x D) np.ndarray
            Array of sizes for the N points in view
        """
        if len(self._indices_view) > 0:
            # Get the point sizes and scale for ndim display
            sizes = (
                self.size[
                    np.ix_(self._indices_view, self._slice_input.displayed)
                ].mean(axis=1)
                * self._view_size_scale
            )

        else:
            # if no points, return an empty list
            sizes = np.array([])
        return sizes

    @property
    def _view_symbol(self) -> np.ndarray:
        """Get the symbols of the points in view

        Returns
        -------
        symbol : (N,) np.ndarray
            Array of symbol strings for the N points in view
        """
        return self.symbol[self._indices_view]

    @property
    def _view_border_width(self) -> np.ndarray:
        """Get the border_width of the points in view

        Returns
        -------
        view_border_width : (N,) np.ndarray
            Array of border_widths for the N points in view
        """
        return self.border_width[self._indices_view]

    @property
    def _view_face_color(self) -> np.ndarray:
        """Get the face colors of the points in view

        Returns
        -------
        view_face_color : (N x 4) np.ndarray
            RGBA color array for the face colors of the N points in view.
            If there are no points in view, returns array of length 0.
        """
        return self.face_color[self._indices_view]

    @property
    def _view_border_color(self) -> np.ndarray:
        """Get the border colors of the points in view

        Returns
        -------
        view_border_color : (N x 4) np.ndarray
            RGBA color array for the border colors of the N points in view.
            If there are no points in view, returns array of length 0.
        """
        return self.border_color[self._indices_view]

    def _reset_editable(self) -> None:
        """Set editable mode based on layer properties."""
        # interaction currently does not work for 2D layers being rendered in 3D
        self.editable = not (
            self.ndim == 2 and self._slice_input.ndisplay == 3
        )

    def _on_editable_changed(self) -> None:
        if not self.editable:
            self.mode = Mode.PAN_ZOOM

    def _get_value(self, position) -> Optional[int]:
        """Index of the point at a given 2D position in data coordinates.

        Parameters
        ----------
        position : tuple
            Position in data coordinates.

        Returns
        -------
        value : int or None
            Index of point that is at the current coordinate if any.
        """
        # Display points if there are any in this slice
        view_data = self._view_data
        selection = None
        if len(view_data) > 0:
            displayed_position = [
                position[i] for i in self._slice_input.displayed
            ]
            # Get the point sizes
            # TODO: calculate distance in canvas space to account for canvas_size_limits.
            # Without this implementation, point hover and selection (and anything depending
            # on self.get_value()) won't be aware of the real extent of points, causing
            # unexpected behaviour. See #3734 for details.
            distances = abs(view_data - displayed_position)
            in_slice_matches = np.all(
                distances <= np.expand_dims(self._view_size, axis=1) / 2,
                axis=1,
            )
            indices = np.where(in_slice_matches)[0]
            if len(indices) > 0:
                selection = self._indices_view[indices[-1]]

        return selection

    def _get_value_3d(
        self,
        start_point: np.ndarray,
        end_point: np.ndarray,
        dims_displayed: List[int],
    ) -> Optional[int]:
        """Get the layer data value along a ray

        Parameters
        ----------
        start_point : np.ndarray
            The start position of the ray used to interrogate the data.
        end_point : np.ndarray
            The end position of the ray used to interrogate the data.
        dims_displayed : List[int]
            The indices of the dimensions currently displayed in the Viewer.

        Returns
        -------
        value : Union[int, None]
            The data value along the supplied ray.
        """
        if (start_point is None) or (end_point is None):
            # if the ray doesn't intersect the data volume, no points could have been intersected
            return None
        plane_point, plane_normal = displayed_plane_from_nd_line_segment(
            start_point, end_point, dims_displayed
        )

        # project the in view points onto the plane
        projected_points, projection_distances = project_points_onto_plane(
            points=self._view_data,
            plane_point=plane_point,
            plane_normal=plane_normal,
        )

        # rotate points and plane to be axis aligned with normal [0, 0, 1]
        rotated_points, rotation_matrix = rotate_points(
            points=projected_points,
            current_plane_normal=plane_normal,
            new_plane_normal=[0, 0, 1],
        )
        rotated_click_point = np.dot(rotation_matrix, plane_point)

        # find the points the click intersects
        distances = abs(rotated_points[:, :2] - rotated_click_point[:2])
        in_slice_matches = np.all(
            distances <= np.expand_dims(self._view_size, axis=1) / 2,
            axis=1,
        )
        indices = np.where(in_slice_matches)[0]

        if len(indices) > 0:
            # find the point that is most in the foreground
            candidate_point_distances = projection_distances[indices]
            closest_index = indices[np.argmin(candidate_point_distances)]
            selection = self._indices_view[closest_index]
        else:
            selection = None
        return selection

    def get_ray_intersections(
        self,
        position: List[float],
        view_direction: np.ndarray,
        dims_displayed: List[int],
        world: bool = True,
    ) -> Union[Tuple[np.ndarray, np.ndarray], Tuple[None, None]]:
        """Get the start and end point for the ray extending
        from a point through the displayed bounding box.

        This method overrides the base layer, replacing the bounding box used
        to calculate intersections with a larger one which includes the size
        of points in view.

        Parameters
        ----------
        position
            the position of the point in nD coordinates. World vs. data
            is set by the world keyword argument.
        view_direction : np.ndarray
            a unit vector giving the direction of the ray in nD coordinates.
            World vs. data is set by the world keyword argument.
        dims_displayed
            a list of the dimensions currently being displayed in the viewer.
        world : bool
            True if the provided coordinates are in world coordinates.
            Default value is True.

        Returns
        -------
        start_point : np.ndarray
            The point on the axis-aligned data bounding box that the cursor click
            intersects with. This is the point closest to the camera.
            The point is the full nD coordinates of the layer data.
            If the click does not intersect the axis-aligned data bounding box,
            None is returned.
        end_point : np.ndarray
            The point on the axis-aligned data bounding box that the cursor click
            intersects with. This is the point farthest from the camera.
            The point is the full nD coordinates of the layer data.
            If the click does not intersect the axis-aligned data bounding box,
            None is returned.
        """
        if len(dims_displayed) != 3:
            return None, None

        # create the bounding box in data coordinates
        bounding_box = self._display_bounding_box_augmented(dims_displayed)

        if bounding_box is None:
            return None, None

        start_point, end_point = self._get_ray_intersections(
            position=position,
            view_direction=view_direction,
            dims_displayed=dims_displayed,
            world=world,
            bounding_box=bounding_box,
        )
        return start_point, end_point

    def _set_view_slice(self):
        """Sets the view given the indices to slice with."""

        # The new slicing code makes a request from the existing state and
        # executes the request on the calling thread directly.
        # For async slicing, the calling thread will not be the main thread.
        request = self._make_slice_request_internal(
            self._slice_input, self._slice_indices
        )
        response = request()
        self._update_slice_response(response)

    def _make_slice_request(self, dims) -> Any:
        """Make a Points slice request based on the given dims and these data."""
        slice_input = self._make_slice_input(
            dims.point, dims.ndisplay, dims.order
        )
        # TODO: [see Image]
        #   For the existing sync slicing, slice_indices is passed through
        # to avoid some performance issues related to the evaluation of the
        # data-to-world transform and its inverse. Async slicing currently
        # absorbs these performance issues here, but we can likely improve
        # things either by caching the world-to-data transform on the layer
        # or by lazily evaluating it in the slice task itself.
        slice_indices = slice_input.data_indices(
            self._data_to_world.inverse, round_index=False
        )
        return self._make_slice_request_internal(slice_input, slice_indices)

    @abstractmethod
    def _make_slice_request_internal(
        self, slice_input: _SliceInput, dims_indices: ArrayLike
    ):
        raise NotImplementedError

    def _update_slice_response(self, response: _PointSliceResponse):
        """Handle a slicing response."""
        self._slice_input = response.dims
        indices = response.indices
        scale = response.scale

        # Update the _view_size_scale in accordance to the self._indices_view setter.
        # If out_of_slice_display is False, scale is a number and not an array.
        # Therefore we have an additional if statement checking for
        # self._view_size_scale being an integer.
        if not isinstance(scale, np.ndarray):
            self._view_size_scale = scale
        elif len(self._shown) == 0:
            self._view_size_scale = np.empty(0, int)
        else:
            self._view_size_scale = scale[self.shown[indices]]

        self._indices_view = np.array(indices, dtype=int)
        # get the selected points that are in view
        self._selected_view = list(
            np.intersect1d(
                np.array(list(self._selected_data)),
                self._indices_view,
                return_indices=True,
            )[2]
        )
        with self.events.highlight.blocker():
            self._set_highlight(force=True)

    def _set_highlight(self, force=False):
        """Render highlights of shapes including boundaries, vertices,
        interaction boxes, and the drag selection box when appropriate.
        Highlighting only occurs in Mode.SELECT.

        Parameters
        ----------
        force : bool
            Bool that forces a redraw to occur when `True`
        """
        # Check if any point ids have changed since last call
        if (
            self.selected_data == self._selected_data_stored
            and self._value == self._value_stored
            and np.all(self._drag_box == self._drag_box_stored)
        ) and not force:
            return
        self._selected_data_stored = copy(self.selected_data)
        self._value_stored = copy(self._value)
        self._drag_box_stored = copy(self._drag_box)

        if self._value is not None or len(self._selected_view) > 0:
            if len(self._selected_view) > 0:
                index = copy(self._selected_view)
                # highlight the hovered point if not in adding mode
                if (
                    self._value in self._indices_view
                    and self._mode == Mode.SELECT
                    and not self._is_selecting
                ):
                    hover_point = list(self._indices_view).index(self._value)
                    if hover_point not in index:
                        index.append(hover_point)
                index.sort()
            else:
                # only highlight hovered points in select mode
                if (
                    self._value in self._indices_view
                    and self._mode == Mode.SELECT
                    and not self._is_selecting
                ):
                    hover_point = list(self._indices_view).index(self._value)
                    index = [hover_point]
                else:
                    index = []

            self._highlight_index = index
        else:
            self._highlight_index = []

        # only display dragging selection box in 2D
        if self._is_selecting:
            if self._drag_normal is None:
                pos = create_box(self._drag_box)
            else:
                pos = _create_box_from_corners_3d(
                    self._drag_box, self._drag_normal, self._drag_up
                )
            pos = pos[[*range(4), 0]]
        else:
            pos = None

        self._highlight_box = pos
        self.events.highlight()

    def _update_thumbnail(self):
        """Update thumbnail with current points and colors."""
        colormapped = np.zeros(self._thumbnail_shape)
        colormapped[..., 3] = 1
        view_data = self._view_data
        if len(view_data) > 0:
            # Get the zoom factor required to fit all data in the thumbnail.
            de = self._extent_data
            min_vals = [de[0, i] for i in self._slice_input.displayed]
            shape = np.ceil(
                [de[1, i] - de[0, i] + 1 for i in self._slice_input.displayed]
            ).astype(int)
            zoom_factor = np.divide(
                self._thumbnail_shape[:2], shape[-2:]
            ).min()

            # Maybe subsample the points.
            if len(view_data) > self._max_points_thumbnail:
                thumbnail_indices = np.random.randint(
                    0, len(view_data), self._max_points_thumbnail
                )
                points = view_data[thumbnail_indices]
            else:
                points = view_data
                thumbnail_indices = self._indices_view

            # Calculate the point coordinates in the thumbnail data space.
            thumbnail_shape = np.clip(
                np.ceil(zoom_factor * np.array(shape[:2])).astype(int),
                1,  # smallest side should be 1 pixel wide
                self._thumbnail_shape[:2],
            )
            coords = np.floor(
                (points[:, -2:] - min_vals[-2:] + 0.5) * zoom_factor
            ).astype(int)
            coords = np.clip(coords, 0, thumbnail_shape - 1)

            # Draw single pixel points in the colormapped thumbnail.
            colormapped = np.zeros((*thumbnail_shape, 4))
            colormapped[..., 3] = 1
            colors = self._face.colors[thumbnail_indices]
            colormapped[coords[:, 0], coords[:, 1]] = colors

        colormapped[..., 3] *= self.opacity
        self.thumbnail = colormapped

    @abstractmethod
    def add(self, coords: ArrayLike) -> None:
        """Adds points at coordinates.

        Parameters
        ----------
        coords : array
            Point or points to add to the layer data.
        """
        raise NotImplementedError

    @abstractmethod
    def remove_selected(self) -> None:
        """Removes selected points if any."""
        raise NotImplementedError

    def _move(
        self,
        selection_indices: Sequence[int],
        position: Sequence[Union[int, float]],
    ) -> None:
        """Move points relative to drag start location.

        Parameters
        ----------
        selection_indices : Sequence[int]
            Integer indices of points to move in self.data
        position : tuple
            Position to move points to in data coordinates.
        """
        if len(selection_indices) > 0:
            selection_indices = list(selection_indices)
            disp = list(self._slice_input.displayed)
            self._set_drag_start(selection_indices, position)
            ixgrid = np.ix_(selection_indices, disp)
            center = self._points_data[ixgrid].mean(axis=0)
            shift = np.array(position)[disp] - center - self._drag_start
            self._move_points(ixgrid, shift)
            self.refresh()
        self.events.data(value=self.data)

    @abstractmethod
    def _move_points(
        self, ixgrid: Tuple[np.ndarray, np.ndarray], shift: np.ndarray
    ) -> None:
        """Move points along a set a coordinates given a shift.

        Parameters
        ----------
        ixgrid : Tuple[np.ndarray, np.ndarray]
            Crossproduct indexing grid of node indices and dimensions, see `np.ix_`
        shift : np.ndarray
            Selected coordinates shift
        """
        raise NotImplementedError

    def _set_drag_start(
        self,
        selection_indices: Sequence[int],
        position: Sequence[Union[int, float]],
        center_by_data: bool = True,
    ) -> None:
        """Store the initial position at the start of a drag event.

        Parameters
        ----------
        selection_indices : set of int
            integer indices of selected data used to index into self.data
        position : Sequence of numbers
            position of the drag start in data coordinates.
        center_by_data : bool
            Center the drag start based on the selected data.
            Used for modifier drag_box selection.
        """
        selection_indices = list(selection_indices)
        dims_displayed = list(self._slice_input.displayed)
        if self._drag_start is None:
            self._drag_start = np.array(position, dtype=float)[dims_displayed]
            if len(selection_indices) > 0 and center_by_data:
                center = self._points_data[
                    np.ix_(selection_indices, dims_displayed)
                ].mean(axis=0)
                self._drag_start -= center

    def get_status(
        self,
        position: Optional[Tuple] = None,
        *,
        view_direction: Optional[np.ndarray] = None,
        dims_displayed: Optional[List[int]] = None,
        world: bool = False,
    ) -> dict:
        """Status message information of the data at a coordinate position.

        # Parameters
        # ----------
        # position : tuple
        #     Position in either data or world coordinates.
        # view_direction : Optional[np.ndarray]
        #     A unit vector giving the direction of the ray in nD world coordinates.
        #     The default value is None.
        # dims_displayed : Optional[List[int]]
        #     A list of the dimensions currently being displayed in the viewer.
        #     The default value is None.
        # world : bool
        #     If True the position is taken to be in world coordinates
        #     and converted into data coordinates. False by default.

        # Returns
        # -------
        # source_info : dict
        #     Dict containing information that can be used in a status update.
        #"""
        if position is not None:
            value = self.get_value(
                position,
                view_direction=view_direction,
                dims_displayed=dims_displayed,
                world=world,
            )
        else:
            value = None

        source_info = self._get_source_info()
        source_info['coordinates'] = generate_layer_coords_status(
            position[-self.ndim :], value
        )

        # if this points layer has properties
        properties = self._get_properties(
            position,
            view_direction=view_direction,
            dims_displayed=dims_displayed,
            world=world,
        )
        if properties:
            source_info['coordinates'] += "; " + ", ".join(properties)

        return source_info

    def _get_tooltip_text(
        self,
        position,
        *,
        view_direction: Optional[np.ndarray] = None,
        dims_displayed: Optional[List[int]] = None,
        world: bool = False,
    ):
        """
        tooltip message of the data at a coordinate position.

        Parameters
        ----------
        position : tuple
            Position in either data or world coordinates.
        view_direction : Optional[np.ndarray]
            A unit vector giving the direction of the ray in nD world coordinates.
            The default value is None.
        dims_displayed : Optional[List[int]]
            A list of the dimensions currently being displayed in the viewer.
            The default value is None.
        world : bool
            If True the position is taken to be in world coordinates
            and converted into data coordinates. False by default.

        Returns
        -------
        msg : string
            String containing a message that can be used as a tooltip.
        """
        return "\n".join(
            self._get_properties(
                position,
                view_direction=view_direction,
                dims_displayed=dims_displayed,
                world=world,
            )
        )

    def _get_properties(
        self,
        position,
        *,
        view_direction: Optional[np.ndarray] = None,
        dims_displayed: Optional[List[int]] = None,
        world: bool = False,
    ) -> list:
        if self.features.shape[1] == 0:
            return []

        value = self.get_value(
            position,
            view_direction=view_direction,
            dims_displayed=dims_displayed,
            world=world,
        )
        # if the cursor is not outside the image or on the background
        if value is None or value > len(self._points_data):
            return []

        return [
            f'{k}: {v[value]}'
            for k, v in self.features.items()
            if k != 'index'
            and len(v) > value
            and v[value] is not None
            and not (isinstance(v[value], float) and np.isnan(v[value]))
        ]


class Points(_BasePoints):
    def __init__(
        self,
        data=None,
        *,
        ndim=None,
        features=None,
        feature_defaults=None,
        properties=None,
        text=None,
        symbol='o',
        size=10,
        border_width=0.05,
        border_width_is_relative=True,
        border_color='dimgray',
        border_color_cycle=None,
        border_colormap='viridis',
        border_contrast_limits=None,
        face_color='white',
        face_color_cycle=None,
        face_colormap='viridis',
        face_contrast_limits=None,
        out_of_slice_display=False,
        n_dimensional=None,
        name=None,
        metadata=None,
        scale=None,
        translate=None,
        rotate=None,
        shear=None,
        affine=None,
        opacity=1,
        blending='translucent',
        visible=True,
        cache=True,
        property_choices=None,
        experimental_clipping_planes=None,
        shading='none',
        canvas_size_limits=(2, 10000),
        antialiasing=1,
        shown=True,
    ) -> None:
        if ndim is None and scale is not None:
            ndim = len(scale)

        data, ndim = fix_data_points(data, ndim)

        # Save the point coordinates
        self._data = np.asarray(data)

        super().__init__(
            data,
            ndim=ndim,
            features=features,
            feature_defaults=feature_defaults,
            properties=properties,
            text=text,
            symbol=symbol,
            size=size,
            border_width=border_width,
            border_width_is_relative=border_width_is_relative,
            border_color=border_color,
            border_color_cycle=border_color_cycle,
            border_colormap=border_colormap,
            border_contrast_limits=border_contrast_limits,
            face_color=face_color,
            face_color_cycle=face_color_cycle,
            face_colormap=face_colormap,
            face_contrast_limits=face_contrast_limits,
            out_of_slice_display=out_of_slice_display,
            n_dimensional=n_dimensional,
            name=name,
            metadata=metadata,
            scale=scale,
            translate=translate,
            rotate=rotate,
            shear=shear,
            affine=affine,
            opacity=opacity,
            blending=blending,
            visible=visible,
            cache=cache,
            property_choices=property_choices,
            experimental_clipping_planes=experimental_clipping_planes,
            shading=shading,
            canvas_size_limits=canvas_size_limits,
            antialiasing=antialiasing,
            shown=shown,
        )

    @property
    def _points_data(self) -> np.ndarray:
        """Spatialy distributed coordinates."""
        return self.data

    @property
    def data(self) -> np.ndarray:
        """(N, D) array: coordinates for N points in D dimensions."""
        return self._data

    @data.setter
    def data(self, data: Optional[np.ndarray]):
        data, _ = fix_data_points(data, self.ndim)
        cur_npoints = len(self._data)
        self._data = data

        # Add/remove property and style values based on the number of new points.
        with self.events.blocker_all(), self._border.events.blocker_all(), self._face.events.blocker_all():
            self._feature_table.resize(len(data))
            self.text.apply(self.features)
            if len(data) < cur_npoints:
                # If there are now fewer points, remove the size and colors of the
                # extra ones
                if len(self._border.colors) > len(data):
                    self._border._remove(
                        np.arange(len(data), len(self._border.colors))
                    )
                if len(self._face.colors) > len(data):
                    self._face._remove(
                        np.arange(len(data), len(self._face.colors))
                    )
                self._shown = self._shown[: len(data)]
                self._size = self._size[: len(data)]
                self._border_width = self._border_width[: len(data)]
                self._symbol = self._symbol[: len(data)]

            elif len(data) > cur_npoints:
                # If there are now more points, add the size and colors of the
                # new ones
                adding = len(data) - cur_npoints
                if len(self._size) > 0:
                    new_size = copy(self._size[-1])
                    for i in self._slice_input.displayed:
                        new_size[i] = self.current_size
                else:
                    # Add the default size, with a value for each dimension
                    new_size = np.repeat(
                        self.current_size, self._size.shape[1]
                    )
                size = np.repeat([new_size], adding, axis=0)

                if len(self._border_width) > 0:
                    new_border_width = copy(self._border_width[-1])
                else:
                    new_border_width = self.current_border_width
                border_width = np.repeat([new_border_width], adding, axis=0)

                if len(self._symbol) > 0:
                    new_symbol = copy(self._symbol[-1])
                else:
                    new_symbol = self.current_symbol
                symbol = np.repeat([new_symbol], adding, axis=0)

                # Add new colors, updating the current property value before
                # to handle any in-place modification of feature_defaults.
                # Also see: https://github.com/napari/napari/issues/5634
                current_properties = self._feature_table.currents()
                self._border._update_current_properties(current_properties)
                self._border._add(n_colors=adding)
                self._face._update_current_properties(current_properties)
                self._face._add(n_colors=adding)

                shown = np.repeat([True], adding, axis=0)
                self._shown = np.concatenate((self._shown, shown), axis=0)

                self.size = np.concatenate((self._size, size), axis=0)
                self.border_width = np.concatenate(
                    (self._border_width, border_width), axis=0
                )
                self.symbol = np.concatenate((self._symbol, symbol), axis=0)
                self.selected_data = set(np.arange(cur_npoints, len(data)))

        self._update_dims()
        self.events.data(value=self.data)
        self._reset_editable()

    def _get_ndim(self) -> int:
        """Determine number of dimensions of the layer."""
        return self.data.shape[1]

    def _make_slice_request_internal(
        self, slice_input: _SliceInput, dims_indices: ArrayLike
    ) -> _PointSliceRequest:
        return _PointSliceRequest(
            dims=slice_input,
            data=self.data,
            dims_indices=dims_indices,
            out_of_slice_display=self.out_of_slice_display,
            size=self.size,
        )

    def add(self, coords):
        """Adds points at coordinates.

        Parameters
        ----------
        coords : array
            Point or points to add to the layer data.
        """
        self.data = np.append(self.data, np.atleast_2d(coords), axis=0)

    def remove_selected(self):
        """Removes selected points if any."""
        index = list(self.selected_data)
        index.sort()
        if len(index):
            self._shown = np.delete(self._shown, index, axis=0)
            self._size = np.delete(self._size, index, axis=0)
            self._symbol = np.delete(self._symbol, index, axis=0)
            self._border_width = np.delete(self._border_width, index, axis=0)
            with self._border.events.blocker_all():
                self._border._remove(indices_to_remove=index)
            with self._face.events.blocker_all():
                self._face._remove(indices_to_remove=index)
            self._feature_table.remove(index)
            self.text.remove(index)
            if self._value in self.selected_data:
                self._value = None
            else:
                if self._value is not None:
                    # update the index of self._value to account for the
                    # data being removed
                    indices_removed = np.array(index) < self._value
                    offset = np.sum(indices_removed)
                    self._value -= offset
                    self._value_stored -= offset

            self.data = np.delete(self.data, index, axis=0)
            self.selected_data = set()

    def _move_points(
        self, ixgrid: Tuple[np.ndarray, np.ndarray], shift: np.ndarray
    ) -> None:
        """Move points along a set a coordinates given a shift.

        Parameters
        ----------
        ixgrid : Tuple[np.ndarray, np.ndarray]
            Crossproduct indexing grid of node indices and dimensions, see `np.ix_`
        shift : np.ndarray
            Selected coordinates shift
        """
        self.data[ixgrid] = self.data[ixgrid] + shift

    def _paste_data(self):
        """Paste any point from clipboard and select them."""
        npoints = len(self._view_data)
        totpoints = len(self._points_data)

        if len(self._clipboard.keys()) > 0:
            not_disp = self._slice_input.not_displayed
            data = deepcopy(self._clipboard['data'])
            offset = [
                self._slice_indices[i] - self._clipboard['indices'][i]
                for i in not_disp
            ]
            data[:, not_disp] = data[:, not_disp] + np.array(offset)
            self._data = np.append(self.data, data, axis=0)
            self._shown = np.append(
                self.shown, deepcopy(self._clipboard['shown']), axis=0
            )
            self._size = np.append(
                self.size, deepcopy(self._clipboard['size']), axis=0
            )
            self._symbol = np.append(
                self.symbol, deepcopy(self._clipboard['symbol']), axis=0
            )

            self._feature_table.append(self._clipboard['features'])

            self.text._paste(**self._clipboard['text'])

            self._border_width = np.append(
                self.border_width,
                deepcopy(self._clipboard['border_width']),
                axis=0,
            )
            self._border._paste(
                colors=self._clipboard['border_color'],
                properties=_features_to_properties(
                    self._clipboard['features']
                ),
            )
            self._face._paste(
                colors=self._clipboard['face_color'],
                properties=_features_to_properties(
                    self._clipboard['features']
                ),
            )

            self._selected_view = list(
                range(npoints, npoints + len(self._clipboard['data']))
            )
            self._selected_data.update(
                set(range(totpoints, totpoints + len(self._clipboard['data'])))
            )
            self.refresh()

    def _copy_data(self):
        """Copy selected points to clipboard."""
        if len(self.selected_data) > 0:
            index = list(self.selected_data)
            self._clipboard = {
                'data': deepcopy(self.data[index]),
                'border_color': deepcopy(self.border_color[index]),
                'face_color': deepcopy(self.face_color[index]),
                'shown': deepcopy(self.shown[index]),
                'size': deepcopy(self.size[index]),
                'symbol': deepcopy(self.symbol[index]),
                'border_width': deepcopy(self.border_width[index]),
                'features': deepcopy(self.features.iloc[index]),
                'indices': self._slice_indices,
                'text': self.text._copy(index),
            }
        else:
            self._clipboard = {}

    def to_mask(
        self,
        *,
        shape: tuple,
        data_to_world: Optional[Affine] = None,
        isotropic_output: bool = True,
    ):
        """Return a binary mask array of all the points as balls.

        Parameters
        ----------
        shape : tuple
            The shape of the mask to be generated.
        data_to_world : Optional[Affine]
            The data-to-world transform of the output mask image. This likely comes from a reference image.
            If None, then this is the same as this layer's data-to-world transform.
        isotropic_output : bool
            If True, then force the output mask to always contain isotropic balls in data/pixel coordinates.
            Otherwise, allow the anisotropy in the data-to-world transform to squash the balls in certain dimensions.
            By default this is True, but you should set it to False if you are going to create a napari image
            layer from the result with the same data-to-world transform and want the visualized balls to be
            roughly isotropic.

        Returns
        -------
        np.ndarray
            The output binary mask array of the given shape containing this layer's points as balls.
        """
        if data_to_world is None:
            data_to_world = self._data_to_world
        mask = np.zeros(shape, dtype=bool)
        mask_world_to_data = data_to_world.inverse
        points_data_to_mask_data = self._data_to_world.compose(
            mask_world_to_data
        )
        points_in_mask_data_coords = np.atleast_2d(
            points_data_to_mask_data(self.data)
        )

        # Calculating the radii of the output points in the mask is complex.

        # Points.size tells the size of the points in pixels in each dimension,
        # so we take the arithmetic mean across dimensions to define a scalar size
        # per point, which is consistent with visualization.
        mean_radii = np.mean(self.size, axis=1, keepdims=True) / 2

        # Scale each radius by the geometric mean scale of the Points layer to
        # keep the balls isotropic when visualized in world coordinates.
        # Then scale each radius by the scale of the output image mask
        # using the geometric mean if isotropic output is desired.
        # The geometric means are used instead of the arithmetic mean
        # to maintain the volume scaling factor of the transforms.
        point_data_to_world_scale = gmean(np.abs(self._data_to_world.scale))
        mask_world_to_data_scale = (
            gmean(np.abs(mask_world_to_data.scale))
            if isotropic_output
            else np.abs(mask_world_to_data.scale)
        )
        radii_scale = point_data_to_world_scale * mask_world_to_data_scale

        output_data_radii = mean_radii * np.atleast_2d(radii_scale)

        for coords, radii in zip(
            points_in_mask_data_coords, output_data_radii
        ):
            # Define a minimal set of coordinates where the mask could be present
            # by defining an inclusive lower and exclusive upper bound for each dimension.
            lower_coords = np.maximum(np.floor(coords - radii), 0).astype(int)
            upper_coords = np.minimum(
                np.ceil(coords + radii) + 1, shape
            ).astype(int)
            # Generate every possible coordinate within the bounds defined above
            # in a grid of size D1 x D2 x ... x Dd x D (e.g. for D=2, this might be 4x5x2).
            submask_coords = [
                range(lower_coords[i], upper_coords[i])
                for i in range(self.ndim)
            ]
            submask_grids = np.stack(
                np.meshgrid(*submask_coords, copy=False, indexing='ij'),
                axis=-1,
            )
            # Update the mask coordinates based on the normalized square distance
            # using a logical or to maintain any existing positive mask locations.
            normalized_square_distances = np.sum(
                ((submask_grids - coords) / radii) ** 2, axis=-1
            )
            mask[np.ix_(*submask_coords)] |= normalized_square_distances <= 1
        return mask
