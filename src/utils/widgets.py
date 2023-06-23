import numpy as np
import ipywidgets as widgets
from IPython.display import display, clear_output
from traitlets import Any

import parameters as params
from algorithms.gp import (
    build_kernel_rbf,
    build_kernel_rbf_periodic,
    build_kernel_matern,
    build_kernel_matern_periodic,
    build_kernel_matern_periodic_approx,
    build_kernel_matern_periodic_truncated,
    build_kernel_matern_periodic_warped,
)
from simulation.objects import EllipseObject, SquareObject, FlowerObject, PolygonObject


class CameraControl(widgets.HBox, widgets.widget_description.DescriptionWidget, widgets.ValueWidget):
    """Custom float slider allowing to scroll and use arrow keys inside readout."""
    value = Any(help="Camera position")

    def __init__(self, **kwargs):
        cam_slider = widgets.FloatSlider(value=0, min=0, max=2*np.pi, step=0.01, description="camera", readout=False)
        cam_text = widgets.BoundedFloatText(value=0, min=0, max=2*np.pi, step=0.01, layout=widgets.Layout(flex="0 0 auto", width="4rem"))
        widgets.link((cam_slider, "value"), (cam_text, "value"))
        widgets.link((cam_slider, "value"), (self, "value"))
        super().__init__(children=[cam_slider, cam_text], layout=widgets.Layout(width="var(--jp-widgets-inline-width)"), **kwargs)


class ObjectSelector(widgets.VBox, widgets.widget_description.DescriptionWidget, widgets.ValueWidget):
    value = Any(help="Selected object")

    def __init__(self, **kwargs):
        # define options
        options = [
            (EllipseObject.name, widgets.interactive(
                EllipseObject.build,
                a=widgets.FloatSlider(value=6, min=params.OBJ_D_MIN, max=params.OBJ_D_MAX, step=0.1),
                b=widgets.FloatSlider(value=6, min=params.OBJ_D_MIN, max=params.OBJ_D_MAX, step=0.1),
            )),
            (SquareObject.name, widgets.interactive(
                SquareObject.build,
                width=widgets.FloatSlider(value=8, min=2*params.OBJ_D_MIN, max=2*params.OBJ_D_MAX/np.sqrt(2), step=0.1),
            )),
            (FlowerObject.name, widgets.interactive(
                FlowerObject.build,
                amplitude=widgets.FloatSlider(value=2, min=0, max=(params.OBJ_D_MAX - params.OBJ_D_MIN) / 2, step=0.1),
                frequency=widgets.IntSlider(value=5, min=0, max=15),
            )),
            (PolygonObject.name, widgets.interactive(
                PolygonObject.build,
                name=widgets.Dropdown(options=PolygonObject.polygon_names, index=1),
            )),
        ]

        # create widget for selecting object
        self.dropdown_objects = widgets.Dropdown(options=options, index=3, description="object")
        self.dropdown_objects.observe(self.__change_obj, names="value")

        # create widget for configuring selected object
        self.out_object_config = widgets.Output()
        with self.out_object_config:
            display(self.dropdown_objects.value)

        # register handler to update widget value
        for _, option in self.dropdown_objects.options:
            if isinstance(option, widgets.interactive):
                for widget in option.children:
                    widget.observe(self.__update_value, names="value")
        
        super().__init__(children=[self.dropdown_objects, self.out_object_config], **kwargs)
        self.__update_value()
    
    # PRIVATE METHODS
    
    def __change_obj(self, change):
        with self.out_object_config:
            if isinstance(change["new"], widgets.interactive):
                display(change["new"], clear=True)
            else:
                clear_output()
        self.__update_value()
    
    def __update_value(self, *args):
        if isinstance(self.dropdown_objects.value, widgets.interactive):
            self.value = self.dropdown_objects.value.result
        else:
            self.value = self.dropdown_objects.value


class KernelSelector(widgets.VBox, widgets.widget_description.DescriptionWidget, widgets.ValueWidget):
    value = Any(help="Selected Object")

    def __init__(self, **kwargs):
        # define options
        options = [
            ("rbf", widgets.interactive(
                build_kernel_rbf,
                sigma=widgets.FloatSlider(value=(params.OBJ_D_MAX - params.OBJ_D_MIN) / 4, min=0, max=10, step=0.1),
                l=widgets.FloatSlider(value=0.2, min=0.02, max=2, step=0.02),
            )),
            ("rbf_periodic", widgets.interactive(
                build_kernel_rbf_periodic,
                sigma=widgets.FloatSlider(value=(params.OBJ_D_MAX - params.OBJ_D_MIN) / 4, min=0, max=10, step=0.1),
                l=widgets.FloatSlider(value=0.2, min=0.02, max=2, step=0.02),
            )),
            ("matern", widgets.interactive(
                build_kernel_matern,
                sigma=widgets.FloatSlider(value=(params.OBJ_D_MAX - params.OBJ_D_MIN) / 4, min=0, max=10, step=0.1),
                l=widgets.FloatSlider(value=0.2, min=0.02, max=2, step=0.02),
                nu=widgets.FloatSlider(value=1.5, min=0.5, max=5, step=0.1),
            )),
            ("matern_periodic", widgets.interactive(
                build_kernel_matern_periodic,
                sigma=widgets.FloatSlider(value=(params.OBJ_D_MAX - params.OBJ_D_MIN) / 4, min=0, max=10, step=0.1),
                l=widgets.FloatSlider(value=0.2, min=0.02, max=2, step=0.02),
                nu=widgets.Dropdown(options=[0.5, 1.5, 2.5], index=1),
                normalized=widgets.fixed(True),
            )),
            ("matern_periodic_approx", widgets.interactive(
                build_kernel_matern_periodic_approx,
                sigma=widgets.FloatSlider(value=(params.OBJ_D_MAX - params.OBJ_D_MIN) / 4, min=0, max=10, step=0.1),
                l=widgets.FloatSlider(value=0.2, min=0.02, max=2, step=0.02),
                nu=widgets.FloatSlider(value=1.5, min=0.5, max=5, step=0.1),
                n_approx=widgets.IntSlider(value=1, min=0, max=10),
            )),
            ("matern_periodic_truncated", widgets.interactive(
                build_kernel_matern_periodic_truncated,
                sigma=widgets.FloatSlider(value=(params.OBJ_D_MAX - params.OBJ_D_MIN) / 4, min=0, max=10, step=0.1),
                l=widgets.FloatSlider(value=0.2, min=0.02, max=2, step=0.02),
                nu=widgets.FloatSlider(value=1.5, min=0.5, max=5, step=0.1),
                c1=widgets.FloatSlider(value=np.pi, min=0, max=2*np.pi),
                c2=widgets.FloatSlider(value=2*np.pi, min=2*np.pi, max=4*np.pi, step=np.pi/4),
                n=widgets.fixed(None),
            )),
            ("matern_periodic_warped", widgets.interactive(
                build_kernel_matern_periodic_warped,
                sigma=widgets.FloatSlider(value=(params.OBJ_D_MAX - params.OBJ_D_MIN) / 4, min=0, max=10, step=0.1),
                l=widgets.FloatSlider(value=0.2, min=0.02, max=2, step=0.02),
                nu=widgets.FloatSlider(value=1.5, min=0.5, max=5, step=0.1),
            )),
        ]

        # create widget for selecting kernel
        self.dropdown_kernels = widgets.Dropdown(options=options, description="kernel", index=3)
        self.dropdown_kernels.observe(self.__change_kernel, names="value")

        # create widget for configuring selected kernel
        self.out_kernel_config = widgets.Output()
        with self.out_kernel_config:
            display(self.dropdown_kernels.value)
        
        # register handler to update widget value
        for _, interactive in self.dropdown_kernels.options:
            for widget in interactive.children:
                widget.observe(self.__update_value, names="value")

        super().__init__(children=[self.dropdown_kernels, self.out_kernel_config], **kwargs)
        self.__update_value()
    
    # PRIVATE METHODS
    
    def __change_kernel(self, change):
            with self.out_kernel_config:
                if isinstance(change["new"], widgets.interactive):
                    display(change["new"], clear=True)
                else:
                    clear_output()
            self.__update_value()
    
    def __update_value(self, *args):
        if isinstance(self.dropdown_kernels.value, widgets.interactive):
            self.value = self.dropdown_kernels.value.result
        else:
            self.value = self.dropdown_kernels.value
