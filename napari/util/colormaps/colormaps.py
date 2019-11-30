import os
from .vendored import colorconv, cm
import numpy as np
from vispy.color import get_colormap, get_colormaps, BaseColormap, Colormap, ColorArray
import vispy

_matplotlib_list_file = os.path.join(
    os.path.dirname(__file__), 'matplotlib_cmaps.txt'
)
with open(_matplotlib_list_file) as fin:
    matplotlib_colormaps = [line.rstrip() for line in fin]


primary_color_names = ['red', 'green', 'blue', 'cyan', 'magenta', 'yellow']
primary_colors = np.array(
    [(1, 0, 0), (0, 1, 0), (0, 0, 1), (0, 1, 1), (1, 0, 1), (1, 1, 0)],
    dtype=float,
)


simple_colormaps = {
    name: Colormap([[0.0, 0.0, 0.0], color])
    for name, color in zip(primary_color_names, primary_colors)
}


def _all_rgb():
    """Return all 256**3 valid rgb tuples."""
    base = np.arange(256, dtype=np.uint8)
    r, g, b = np.meshgrid(base, base, base, indexing='ij')
    return np.stack((r, g, b), axis=-1).reshape((-1, 3))


# obtained with colorconv.rgb2luv(_all_rgb().reshape((-1, 256, 3)))
LUVMIN = np.array([0.0, -83.07790815, -134.09790293])
LUVMAX = np.array([100.0, 175.01447356, 107.39905336])
LUVRNG = LUVMAX - LUVMIN

# obtained with colorconv.rgb2lab(_all_rgb().reshape((-1, 256, 3)))
LABMIN = np.array([0.0, -86.18302974, -107.85730021])
LABMAX = np.array([100.0, 98.23305386, 94.47812228])
LABRNG = LABMAX - LABMIN


def _validate_rgb(colors, *, tolerance=0.0):
    """Return the subset of colors that is in [0, 1] for all channels.

    Parameters
    ----------
    colors : array of float, shape (N, 3)
        Input colors in RGB space.

    Other Parameters
    ----------------
    tolerance : float, optional
        Values outside of the range by less than ``tolerance`` are allowed and
        clipped to be within the range.

    Returns
    -------
    filtered_colors : array of float, shape (M, 3), M <= N
        The subset of colors that are in valid RGB space.

    Examples
    --------
    >>> colors = np.array([[  0. , 1.,  1.  ],
    ...                    [  1.1, 0., -0.03],
    ...                    [  1.2, 1.,  0.5 ]])
    >>> _validate_rgb(colors)
    array([[0., 1., 1.]])
    >>> _validate_rgb(colors, tolerance=0.15)
    array([[0., 1., 1.],
           [1., 0., 0.]])
    """
    lo = 0 - tolerance
    hi = 1 + tolerance
    valid = np.all((colors > lo) & (colors < hi), axis=1)
    filtered_colors = np.clip(colors[valid], 0, 1)
    return filtered_colors


def _low_discrepancy_image(image, seed=0.5):
    """Generate a 1d low discrepancy sequence of coordinates.

    Parameters
    ----------
    labels : array of int
        A set of labels or label image.
    seed : float
        The seed from which to start the quasirandom sequence.

    Returns
    -------
    image_out : array of float
        The set of ``labels`` remapped to [0, 1] quasirandomly.

    """
    phi = 1.6180339887498948482
    image_out = (seed + image / phi) % 1
    # Clipping slightly above 0 and below 1 is necessary to ensure that the
    # labels do not get mapped to 0 which is represented by the background
    # and is transparent
    return np.clip(image_out, 0.00001, 1.0 - 0.00001)


def _low_discrepancy(dim, n, seed=0.5):
    """Generate a 1d, 2d, or 3d low discrepancy sequence of coordinates.

    Parameters
    ----------
    dim : one of {1, 2, 3}
        The dimensionality of the sequence.
    n : int
        How many points to generate.
    seed : float or array of float, shape (dim,)
        The seed from which to start the quasirandom sequence.

    Returns
    -------
    pts : array of float, shape (n, dim)
        The sampled points.

    References
    ----------
    ..[1]: http://extremelearning.com.au/unreasonable-effectiveness-of-quasirandom-sequences/  # noqa: E501
    """
    phi1 = 1.6180339887498948482
    phi2 = 1.32471795724474602596
    phi3 = 1.22074408460575947536
    seed = np.broadcast_to(seed, (1, dim))
    phi = np.array([phi1, phi2, phi3])
    g = 1 / phi
    n = np.reshape(np.arange(n), (n, 1))
    pts = (seed + (n * g[:dim])) % 1
    return pts


def _color_random(n, *, colorspace='lab', tolerance=0.0, seed=0.5):
    """Generate n random RGB colors uniformly from LAB or LUV space.

    Parameters
    ----------
    n : int
        Number of colors to generate.
    colorspace : str, one of {'lab', 'luv', 'rgb'}
        The colorspace from which to get random colors.
    tolerance : float
        How much margin to allow for out-of-range RGB values (these are
        clipped to be in-range).
    seed : float or array of float, shape (3,)
        Value from which to start the quasirandom sequence.

    Returns
    -------
    rgb : array of float, shape (n, 3)
        RGB colors chosen uniformly at random from given colorspace.
    """
    factor = 6  # about 1/5 of random LUV tuples are inside the space
    expand_factor = 2
    rgb = np.zeros((0, 3))
    while len(rgb) < n:
        random = _low_discrepancy(3, n * factor, seed=seed)
        if colorspace == 'luv':
            raw_rgb = colorconv.luv2rgb(random * LUVRNG + LUVMIN)
        elif colorspace == 'rgb':
            raw_rgb = random
        else:  # 'lab' by default
            raw_rgb = colorconv.lab2rgb(random * LABRNG + LABMIN)
        rgb = _validate_rgb(raw_rgb, tolerance=tolerance)
        factor *= expand_factor
    return rgb[:n]


def label_colormap(num_colors=256, seed=0.5):
    """Produce a colormap suitable for use with a given label set.

    Parameters
    ----------
    num_colors : int, optional
        Number of unique colors to use. Default used if not given.
    seed : float or array of float, length 3
        The seed for the random color generator.

    Returns
    -------
    cmap : vispy.color.Colormap
        A colormap for use with labels are remapped to [0, 1].

    Notes
    -----
    0 always maps to fully transparent.
    """
    # Starting the control points slightly above 0 and below 1 is necessary
    # to ensure that the background pixel 0 is transparent
    midpoints = np.linspace(0.00001, 1 - 0.00001, num_colors - 1)
    control_points = np.concatenate(([0], midpoints, [1.0]))
    # make sure to add an alpha channel to the colors
    colors = np.concatenate(
        (_color_random(num_colors, seed=seed), np.full((num_colors, 1), 1)),
        axis=1,
    )
    colors[0, :] = 0  # ensure alpha is 0 for label 0
    cmap = Colormap(
        colors=colors, controls=control_points, interpolation='zero'
    )
    return cmap


def vispy_or_mpl_colormap(name):
    """Try to get a colormap from vispy, or convert an mpl one to vispy format.

    Parameters
    ----------
    name : str
        The name of the colormap.

    Returns
    -------
    cmap : vispy.color.Colormap
        The found colormap.

    Raises
    ------
    KeyError
        If no colormap with that name is found within vispy or matplotlib.
    """
    vispy_cmaps = get_colormaps()
    if name in vispy_cmaps:
        cmap = get_colormap(name)
    else:
        try:
            mpl_cmap = getattr(cm, name)
        except AttributeError:
            raise KeyError(
                f'Colormap "{name}" not found in either vispy '
                'or matplotlib.'
            )
        mpl_colors = mpl_cmap(np.linspace(0, 1, 256))
        cmap = Colormap(mpl_colors)
    return cmap


# Fire and Grays are two colormaps that work well for
# translucent and additive volume rendering - add
# them to best_3d_colormaps, append them to
# all the existing colormaps


class TransFire(BaseColormap):
    glsl_map = """
    vec4 translucent_fire(float t) {
        return vec4(pow(t, 0.5), t, t*t, max(0, t*1.05 - 0.05));
    }
    """

    def map(self, t):
        if isinstance(t, np.ndarray):
            return np.hstack(
                [np.power(t, 0.5), t, t * t, np.maximum(0, t * 1.05 - 0.05)]
            ).astype(np.float32)
        else:
            return np.array(
                [np.power(t, 0.5), t, t * t, np.maximum(0, t * 1.05 - 0.05)],
                dtype=np.float32,
            )


class TransGrays(BaseColormap):
    glsl_map = """
    vec4 translucent_grays(float t) {
        return vec4(t, t, t, t*0.5);
    }
    """

    def map(self, t):
        if isinstance(t, np.ndarray):
            return np.hstack([t, t, t, t * 0.5]).astype(np.float32)
        else:
            return np.array([t, t, t, t * 0.5], dtype=np.float32)


colormaps_3D = {"fire": TransFire(), "gray_trans": TransGrays()}
colormaps_3D = {k: v for k, v in sorted(colormaps_3D.items())}


# A dictionary mapping names to VisPy colormap objects
ALL_COLORMAPS = {k: vispy_or_mpl_colormap(k) for k in matplotlib_colormaps}
ALL_COLORMAPS.update(simple_colormaps)
ALL_COLORMAPS.update(colormaps_3D)

# ... sorted alphabetically by name
AVAILABLE_COLORMAPS = {k: v for k, v in sorted(ALL_COLORMAPS.items())}

# curated colormap sets
# these are selected to look good or at least reasonable when using additive
# blending of multiple channels.
MAGENTA_GREEN = ['magenta', 'green']
RGB = ['red', 'green', 'blue']
CYMRGB = ['cyan', 'yellow', 'magenta', 'red', 'green', 'blue']


# Label colormaps


class LabelColormap(BaseColormap):
    def __init__(self, zero_is_black=True, hue_start=0, saturation=1.0, value=0.7, ncolors=32):
        self.zero_is_black = zero_is_black
        self.ncolors = 32
        
        # Set colors (based on vispy.colormap._HSL)
        hues = np.linspace(0, 360, ncolors + 1)[:-1]
        hues += hue_start
        hues %= 360
        colors = ColorArray([(hue, saturation, value) for hue in hues], color_space='hsv')
        
        self.texture_map_data = np.zeros((ncolors, 1, 4), dtype=np.float32)
        LUT = self.texture_map_data
        c_rgba = ColorArray(colors)._rgba
        colors_rgba = ColorArray(colors[:])._rgba
        LUT[:, 0, :] = colors_rgba

        self.glsl_map_base = """
        uniform sampler2D texture2D_LUT;
        vec4 random(float t) {
            $zero_is_black
            return texture2D(texture2D_LUT, vec2(0.0, t));
        }
        """
        self.update_shader()
        super().__init__()

    def update_shader(self):
        self.glsl_map = self.glsl_map_base
        if (self.zero_is_black):
            self.glsl_map = self.glsl_map.replace('$zero_is_black', 'if (t == 0.0) { return vec4(0.0, 0.0, 0.0, 0.0); }')
        else:
            self.glsl_map = self.glsl_map.replace('$zero_is_black', '')

    def map(self, t):
        t_index = np.mod((t * self.ncolors).astype(np.int), self.ncolors)
        r = self.texture_map_data[t_index,0,0]
        g = self.texture_map_data[t_index,0,1]
        b = self.texture_map_data[t_index,0,2]
        colors = np.concatenate((r, g, b, np.ones_like(b)), axis=1)
        if self.zero_is_black:
            return np.where(t == 0, np.zeros_like(colors), colors)
        else:
            return colors

    def texture_lut(self):
        # Called from vispy.visuals.image
        texture_LUT = vispy.gloo.Texture2D(np.zeros(self.texture_map_data.shape), wrapping='repeat')
        texture_LUT.set_data(self.texture_map_data, offset=None, copy=True)
        return texture_LUT