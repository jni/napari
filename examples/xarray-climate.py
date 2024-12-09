"""Viewing thredds wr45/ops_aps3 data in napari.

Download bash commands:

# download model prediction data
curl -O https://thredds.nci.org.au/thredds/fileServer/wr45/ops_aps3/access-g/1/20241104/0000/fc/ml/air_temp.nc
curl -O https://thredds.nci.org.au/thredds/fileServer/wr45/ops_aps3/access-g/1/20241104/0000/fc/ml/spec_hum.nc

# download corresponding 10 days' worth of measurements
mkdir an && cd an  # use 'an' folder for single time points
for day in 04 05 06 07 08 09 10 11 12 13; do
  for hour in 00 06 12 18; do
    curl https://thredds.nci.org.au/thredds/fileServer/wr45/ops_aps3/access-g/1/202411${day}/${hour}00/an/ml/spec_hum.nc -o ${day}-${hour}-spec_hum.nc
    curl https://thredds.nci.org.au/thredds/fileServer/wr45/ops_aps3/access-g/1/202411${day}/${hour}00/an/ml/air_temp.nc -o ${day}-${hour}-air_temp.nc
  done
done
"""

from glob import glob
from pathlib import Path

import numpy as np
import xarray as xr

import napari

root_dir = Path('/Users/jni/data/thredds/20241104')

def get_scale_translate(dataset, array_name, invert_lat=False):
    """Get the translate/offset and scale parameters for an xarray dataset.

    This code assumes that the dataset is regularly spaced. You should
    interpolate your data if it is sampled at irregular spaces.

    Parameters
    ----------
    dataset : xr.Dataset
        The dataset containing the array to be displayed.
    array_name : str
        The name of the xarray DataArray within `dataset` to be displayed in
        napari.
    invert_lat : bool
        Whether to invert the latitude values.

        napari's axes follow zyx convention, with y pointing down.

        latitude decreases going down conventionally (putting North at the top
        and having southern latitudes be negative), so here we multiply by -1
        to display the globe conventionally. Unfortunately, this means the
        coordinates displayed by the viewer on hover will show northern
        latitudes as negative. The true fix is to add a transformation
        between the world/scene coordinates and the canvas in the napari code
        base, but that might take some time. An alternative for now is to use
        private APIs to find the VisPy camera and change its orientation, like
        so:

        viewer.window._qt_window._qt_viewer.canvas.camera._2D_camera.flip = (0, 0, 0)
    """
    array = getattr(dataset, array_name)
    if array is None:
        raise ValueError(f'{dataset} has no array with name {array_name}')
    dims = [getattr(dataset, dim) for dim in array.dims]
    translate = [float(d[0]) for d in dims]
    scale = [float(d[1] - d[0]) for d in dims]
    if invert_lat:
        lat_pos, lat_name = next(
                (i, dim)
                for i, dim in enumerate(array.dims)
                if dim.startswith('lat')
                )
        scale[lat_pos] *= -1
        translate[lat_pos] = float(getattr(dataset, lat_name)[-1])
    return {'scale': scale, 'translate': translate}


# open the model dataset
ds = xr.open_dataset(
        root_dir / 'spec_hum.nc',
        chunks={'time': 1, 'theta_lvl': 1},
        )

# Show the raw (not resampled) model data
viewer, model_layer = napari.imshow(
        ds.spec_hum,
        name='model',
        **get_scale_translate(ds, 'spec_hum'),
        )
viewer.dims.axis_labels = ds.spec_hum.dims
# currently no private API to flip the camera, so increasing y is up,
# so we use these private attributes.
viewer.window._qt_window._qt_viewer.canvas.camera._2D_camera.flip = (0, 0, 0)

# open the measurement data
an = xr.open_mfdataset(sorted(glob(str(root_dir / 'an/*spec_hum.nc'))))

# resample the model data to have even spacing in time
start, stop, step = [
        np.array(elem)[()]
        for elem in (ds.time[0], ds.time[-1], ds.time[1] - ds.time[0])
        ]
ds_reg = ds.interp(
        coords={'time': np.arange(start, stop, step)},
        method='nearest',
        assume_sorted=True,
        )

# show the resampled model data overlaid on the measurement data
# note: this has performance issues due to the live resampling.
viewer2 = napari.Viewer()
model = viewer2.add_image(
        ds_reg.spec_hum,
        name='model',
        **get_scale_translate(ds_reg, 'spec_hum'),
        colormap='magenta',
        )
gt = viewer2.add_image(
        an.spec_hum,
        name='measurement',
        **get_scale_translate(an, 'spec_hum'),
        colormap='green',
        blending='additive',
        )
viewer2.dims.axis_labels = ds_reg.spec_hum.dims
# currently no private API to flip the camera, so increasing y is up,
# so we use these private attributes.
viewer2.window._qt_window._qt_viewer.canvas.camera._2D_camera.flip = (0, 0, 0)

if __name__ == '__main__':
    napari.run()