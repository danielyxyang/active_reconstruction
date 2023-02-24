import numpy as np
import ipywidgets as widgets
from IPython.display import display, clear_output
from traitlets import Any

from parameters import GRID_H, OBJ_D_MIN, OBJ_D_MAX
from algorithms.gp import (
    build_kernel_rbf,
    build_kernel_periodic,
    build_kernel_matern,
    build_kernel_matern_periodic,
)
from simulation.objects import EllipseObject, SquareObject, FlowerObject, PolygonObject


def build_widget_cam_slider():
    return widgets.FloatSlider(value=0, min=0, max=2*np.pi, step=0.01, description="camera")


def build_widget_outputs(names):
    return {name: widgets.Output() for name in names}


class ObjectSelector(widgets.VBox, widgets.widget_description.DescriptionWidget, widgets.ValueWidget):
    value = Any(help="Selected object")

    def __init__(self, **kwargs):
        # define options
        options = [
            (EllipseObject.name, widgets.interactive(
                EllipseObject.build,
                a=widgets.FloatSlider(value=6, min=OBJ_D_MIN, max=OBJ_D_MAX, step=0.1),
                b=widgets.FloatSlider(value=6, min=OBJ_D_MIN, max=OBJ_D_MAX, step=0.1),
            )),
            (SquareObject.name, widgets.interactive(
                SquareObject.build,
                width=widgets.FloatSlider(value=8, min=2*OBJ_D_MIN, max=2*OBJ_D_MAX/np.sqrt(2), step=0.1),
            )),
            (FlowerObject.name, widgets.interactive(
                FlowerObject.build,
                amplitude=widgets.FloatSlider(value=2, min=0, max=(OBJ_D_MAX - OBJ_D_MIN) / 2, step=0.1),
                frequency=widgets.IntSlider(value=5, min=0, max=15),
            )),
            *[(str(obj), obj) for obj in PolygonObject.build_polygons()]
        ]

        # create widget for selecting object
        self.dropdown_objects = widgets.Dropdown(options=options, description="object")
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
                sigma=widgets.FloatSlider(value=(OBJ_D_MAX - OBJ_D_MIN) / 4, min=0, max=10, step=0.1),
                l=widgets.FloatSlider(value=GRID_H, min=GRID_H / OBJ_D_MAX, max=2, step=GRID_H / OBJ_D_MAX),
            )),
            ("periodic", widgets.interactive(
                build_kernel_periodic,
                sigma=widgets.FloatSlider(value=(OBJ_D_MAX - OBJ_D_MIN) / 4, min=0, max=10, step=0.1),
                l=widgets.FloatSlider(value=GRID_H, min=GRID_H / OBJ_D_MAX, max=2, step=GRID_H / OBJ_D_MAX),
            )),
            ("matern", widgets.interactive(
                build_kernel_matern,
                sigma=widgets.FloatSlider(value=(OBJ_D_MAX - OBJ_D_MIN) / 4, min=0, max=10, step=0.1),
                l=widgets.FloatSlider(value=GRID_H, min=GRID_H / OBJ_D_MAX, max=2, step=GRID_H / OBJ_D_MAX),
                nu=widgets.FloatSlider(value=1.5, min=0.5, max=5, step=0.1),
            )),
            ("matern_periodic", widgets.interactive(
                build_kernel_matern_periodic,
                sigma=widgets.FloatSlider(value=(OBJ_D_MAX - OBJ_D_MIN) / 4, min=0, max=10, step=0.1),
                l=widgets.FloatSlider(value=GRID_H, min=GRID_H / OBJ_D_MAX, max=2, step=GRID_H / OBJ_D_MAX),
                nu=widgets.FloatSlider(value=1.5, min=0.5, max=5, step=0.1),
                n_approx=widgets.IntSlider(value=1, min=0, max=10),
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


class CheckboxList(widgets.VBox, widgets.widget_description.DescriptionWidget, widgets.ValueWidget):
    value = Any(help="Selected checkboxes")

    def __init__(self, options, value=[], description=None, **kwargs):
        self.options = options
        self.checkboxes = {}
        # create description label
        label = [widgets.Label(value=description)] if description is not None else []
        # create checkboxes
        for name in options:
            # value = value[name] if name in value.keys() else True
            self.checkboxes[name] = widgets.Checkbox(value=name in value, indent=False, layout=widgets.Layout(width="auto", margin="0 0 0 20px"), description=name) 
        checkbox_list = [widgets.VBox([self.checkboxes[name] for name in options], **kwargs)]
        # register handler to update widget value
        for checkbox in self.checkboxes.values():
            checkbox.observe(self.__update_value, names="value")

        super().__init__(children=label + checkbox_list)
        self.__update_value()
    
    # PRIVATE METHODS
    
    def __update_value(self, *args):
        self.value = {name: self.checkboxes[name].value for name in self.options}
