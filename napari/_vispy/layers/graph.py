import numpy as np
from vispy import gloo

from napari._vispy.layers.points import VispyPointsLayer
from napari._vispy.visuals.graph import GraphVisual


class VispyGraphLayer(VispyPointsLayer):
    _visual = GraphVisual

    def __init__(self, layer) -> None:
        super().__init__(layer)
        self.layer.events.edges_visible.connect(
            self._on_edge_visibility_change
        )

    def _on_data_change(self):
        self._set_graph_edges_data()
        super()._on_data_change()

    def _set_graph_edges_data(self):
        """Sets the LineVisual (subvisual[4]) with the graph edges data"""
        subvisual = self.node._subvisuals[4]
        edges = self.layer._view_edges_coordinates

        if len(edges) == 0:
            subvisual.visible = False
            return

        subvisual.visible = True
        flat_edges = edges.reshape((-1, edges.shape[-1]))  # (N x 2, D)
        flat_edges = flat_edges[:, ::-1]

        # clearing up buffer, there was a vispy error otherwise
        subvisual._line_visual._pos_vbo = gloo.VertexBuffer()
        subvisual.set_data(
            flat_edges,
            color='white',
            width=1,
        )

    def _on_edge_visibility_change(self):
        """Set the edge visibility by changing some edge colors to alpha=0."""
        subvisual = self.node._subvisuals[4]
        n_edges = self.layer.data.n_edges
        visibility = np.broadcast_to(self.layer.edges_visible, n_edges)
        color = np.ones((n_edges * 2, 4), dtype=np.float32)
        color[:, 3] *= np.repeat(visibility, 2)
        edges = self.layer._view_edges_coordinates
        flat_edges = edges.reshape((-1, edges.shape[-1]))[:, ::-1]
        subvisual.set_data(
            flat_edges,
            color=color,
            width=1,
        )
