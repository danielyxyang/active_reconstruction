import contextlib
import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection, LineCollection
from matplotlib.patches import Circle, Rectangle, Wedge, Polygon
from IPython.display import display

from parameters import GRID_H, OBJ_D_MIN, OBJ_D_MAX, CAM_D, CAM_FOV_RAD, CAM_DOF
from utils.helpers import polar_to_cartesian, cartesian_to_polar, polar_to_pixel


class Plotter():
    interactive = False

    @staticmethod
    def set_interactive(interactive=True):
        Plotter.interactive = interactive

    def __init__(self, mode=None, **kwargs):
        self.mode = mode
        self.colors = {
            "grid": "gray",
            "camera": "blue",
            "object": "red",
            "confidence": "gray",
            "confidence_lower": "dimgray",
            "confidence_upper": "darkgray",
        }
        self.fig = None
        self.axis = None # currently active axis
        self.axes = {} # dictionary of all axes
        self.__axes_context = [] # stack storing previously active axis for contextmanagers
        self.artists = {}
        self.__displayed = False

        self.create_plot(**kwargs)
    
    def __del__(self):
        if self.fig is not None:
            plt.close(self.fig)

    # HELPER METHODS

    def static(self, key, plt_f, visible=True):
        # display artist
        if key not in self.artists.keys():
            self.artists[key] = plt_f()
        # set visibility of artist
        self.set_visible(self.artists[key], visible)

    def dynamic(self, key, plt_f, update_f, visible=True):
        # display or update artist
        if key not in self.artists.keys():
            self.artists[key] = plt_f()
        else:
            update_f(self.artists[key])
        # set visibility of artist
        self.set_visible(self.artists[key], visible)
    
    def dynamic_plot(self, key, *args, visible=True, **kwargs):
        self.dynamic(
            key,
            lambda: self.axis.plot(*args, **kwargs),
            lambda lines: [line.set_data(*args[2*i:2*i+2]) for i, line in enumerate(lines)],
            visible=visible,
        )
    
    def dynamic_patch_collection(self, key, patches, visible=True, **kwargs):
        self.dynamic(
            key,
            lambda: self.axis.add_collection(PatchCollection(patches, **kwargs)),
            lambda collection: collection.set_paths(patches),
            visible=visible,
        )

    def set_visible(self, item, visible):
        if isinstance(item, plt.Artist):
            item.set_visible(visible)
        elif isinstance(item, list):
            for subitem in item:
                subitem.set_visible(visible)
        else:
            print("WARNING: not able to change visibility of {}".format(item))

    def args_scatter(self, size):
        return dict(linestyle="none", marker="o", markersize=size)

    def args_to_edgecolor(self, kwargs):
        kwargs.update(dict(facecolor="none", edgecolor=kwargs["color"]))
        kwargs.pop("color")
        return kwargs
    
    def args_to_facecolor(self, kwargs):
        kwargs.update(dict(facecolor=kwargs["color"], edgecolor="none"))
        kwargs.pop("color")
        return kwargs

    def highlight_pixels(self, surface_points):
        pixels = polar_to_pixel(surface_points[0], surface_points[1])
        return [Rectangle((GRID_H * (px-0.5), GRID_H * (py-0.5)), GRID_H, GRID_H) for px, py in pixels.T]


    # PUBLIC METHODS

    def create_plot(self, figsize=None, title=None, xlabel=None, ylabel=None, rlim=None):
        if rlim is None:
            rlim = CAM_D * 1.1
        if np.isscalar(figsize):
            figsize = (figsize, figsize) # assume quadratic figure
        
        # create figure
        if self.fig is not None:
            plt.close(self.fig)
        self.fig, self.axis = plt.subplots(figsize=figsize, constrained_layout=True)
        self.axes = {"main": self.axis}
        self.__axes_context = []
        self.artists = {}
        self.require_display = True

        # configure figure canvas for ipympl
        if Plotter.interactive:
            self.fig.canvas.toolbar_position = "top"
            self.fig.canvas.header_visible = False

        # set mode-specific settings
        if self.mode == "world":
            # set descriptions
            self.axis.set_title("World" if title is None else title)
            self.axis.set_xlabel("x coordinates [m]" if xlabel is None else xlabel)
            self.axis.set_ylabel("y coordinates [m]" if ylabel is None else ylabel)
            # set aspect ratio
            self.axis.set_aspect("equal")
            # set view limits
            self.axis.set_xlim([-rlim, rlim])
            self.axis.set_ylim([-rlim, rlim])
        elif self.mode == "polar":
            # set descriptions
            self.axis.set_title("World in Polar Representation" if title is None else title)
            self.axis.set_xlabel("polar angle [rad]" if xlabel is None else xlabel)
            self.axis.set_ylabel("radial distance [m]" if ylabel is None else ylabel)
            # set view limits
            self.axis.set_xlim([0, 2*np.pi])
            self.axis.set_ylim([0, rlim])
        else:
            # set descriptions
            self.axis.set_title(title)
            self.axis.set_xlabel(xlabel)
            self.axis.set_ylabel(ylabel)
        
    def reset(self):
        # remove all artists
        self.artists = {}
        # clear the main axis and remove all secondary axes
        for name, axis in self.axes.items():
            if name == "main":
                axis.clear()
            else:
                axis.remove()

    def display(self, out=None, clear=True, rescale=False):
        if out is None:
            out = contextlib.nullcontext()
        
        if Plotter.interactive:
            if self.__displayed:
                # (optional) rescale plot automatically
                if rescale:
                    self.axis.relim()
                    self.axis.autoscale(tight=True)
                # redraw plot
                self.fig.canvas.draw()
                self.fig.canvas.flush_events()
            else:
                # display plot
                with out:
                    display(self.fig.canvas, clear=clear)
                self.__displayed = True
        else:
            with out:
                display(self.fig, clear=clear)
            self.__displayed = True

    @contextlib.contextmanager
    def use_axis(self, name, ylabel=None):
        """Create context manager for additional y-axes sharing the same x-axis."""
        if name not in self.axes.keys():
            self.axes[name] = self.axes["main"].twinx()

        self.__axes_context.append(self.axis)
        self.axis = self.axes[name]
        self.axis.set_ylabel(ylabel)
        try:
            yield self
        finally:
            self.axis = self.__axes_context.pop()

    def plot_grid(self, show_grid=True):
        color = self.colors["grid"]

        kwargs = dict(color=color, linestyle="-", linewidth=0.2)
        # TODO improve
        if self.mode == "world":
            # kwargs = dict(visible=True, **kwargs) if show else dict(visible=False)
            # grid_range_x = np.arange(*self.axis.get_xlim(), GRID_H) + GRID_H/2
            # grid_range_y = np.arange(*self.axis.get_ylim(), GRID_H) + GRID_H/2
            # self.axis.set_xticks(grid_range_x, minor=True)
            # self.axis.set_yticks(grid_range_y, minor=True)
            # self.axis.tick_params(which="minor", bottom=False, left=False)
            # self.axis.grid(which="minor", **kwargs)
            def plot_f():
                grid_range_x = np.arange(*self.axis.get_xlim(), GRID_H) + GRID_H/2
                grid_range_y = np.arange(*self.axis.get_ylim(), GRID_H) + GRID_H/2
                horizontal_lines = np.stack([
                    np.array([np.full_like(grid_range_y, grid_range_x[0]), grid_range_y]).T,
                    np.array([np.full_like(grid_range_y, grid_range_x[-1]), grid_range_y]).T,
                ], axis=-2)
                vertical_lines = np.stack([
                    np.array([grid_range_x, np.full_like(grid_range_x, grid_range_y[0])]).T,
                    np.array([grid_range_x, np.full_like(grid_range_x, grid_range_y[-1])]).T,
                ], axis=-2)
                lines = np.concatenate([horizontal_lines, vertical_lines])
                return self.axis.add_collection(LineCollection(lines, **kwargs))
            self.static("plot_grid", plot_f, visible=show_grid)
        elif self.mode == "polar":
            # TODO some strange artefacts???
            def plot_f():
                rlim = self.axis.get_ylim()[1]
                grid_range = np.arange(-rlim, rlim, GRID_H) + GRID_H/2
                horizontal1 = np.array([np.full_like(grid_range, -rlim), grid_range]).T
                horizontal2 = np.array([np.full_like(grid_range, rlim), grid_range]).T
                horizontal_lines = np.linspace(horizontal1, horizontal2, num=100, axis=1)
                grid_range_pos = grid_range[grid_range >= 0]
                grid_range_neg = grid_range[grid_range < 0]
                vertical1 = np.array([grid_range_pos, np.full_like(grid_range_pos, -rlim)]).T
                vertical2 = np.array([grid_range_pos, np.full_like(grid_range_pos, 0)]).T
                vertical3 = np.array([grid_range_pos, np.full_like(grid_range_pos, rlim)]).T
                vertical4 = np.array([grid_range_neg, np.full_like(grid_range_neg, -rlim)]).T
                vertical5 = np.array([grid_range_neg, np.full_like(grid_range_neg, rlim)]).T
                vertical_lines1 = np.linspace(vertical1, vertical2, num=100, endpoint=False, axis=1)
                vertical_lines2 = np.linspace(vertical2, vertical3, num=100, axis=1)
                vertical_lines3 = np.linspace(vertical4, vertical5, num=100, axis=1)
                lines = np.concatenate([horizontal_lines, vertical_lines1, vertical_lines2, vertical_lines3])
                lines = cartesian_to_polar(*lines.T).T
                lines[:, :, 0] %= 2*np.pi
                return self.axis.add_collection(LineCollection(lines, **kwargs))
            self.static("plot_grid", plot_f, visible=show_grid)

    def plot_camera(self, camera, show_camera=True, show_los=True, show_fov=True, show_view_circle=True, color=None, name=None):
        if color is None:
            color = self.colors["camera"]
        
        x, y = polar_to_cartesian(camera.theta, CAM_D)

        # plot camera
        key = "plot_camera:{}:camera".format(name)
        kwargs = dict(**self.args_scatter(3), color=color, label="Camera")
        if self.mode == "world":
            self.dynamic_plot(key, x, y, **kwargs, visible=show_camera)
        elif self.mode == "polar":
            self.dynamic_plot(key, camera.theta, CAM_D, **kwargs, visible=show_camera)

        # plot line of sight
        key = "plot_camera:{}:los".format(name)
        kwargs = dict(color=color, linestyle="-", linewidth=1, alpha=0.5)
        length = CAM_DOF if show_los != "position" else CAM_D
        if self.mode == "world":
            x_dof, y_dof = polar_to_cartesian(*camera.camera_to_polar(0, length))
            self.dynamic_plot(key, [x, x_dof], [y, y_dof], **kwargs, visible=show_los)
        elif self.mode == "polar":
            if length <= CAM_D:
                lines = [[camera.theta, camera.theta], [CAM_D, CAM_D - length], [], []]
            else:
                lines = [[camera.theta, camera.theta], [CAM_D, 0], [(camera.theta + np.pi) % (2*np.pi), (camera.theta + np.pi) % (2*np.pi)], [0, length - CAM_D]]
            self.dynamic_plot(key, *lines, **kwargs, visible=show_los)

        # plot field of view
        key = "plot_camera:{}:fov".format(name)
        kwargs_boundary = dict(color=color, alpha=0.4)
        kwargs_region = dict(color=color, alpha=0.1)
        beta1 = CAM_FOV_RAD / 2
        beta2 = -CAM_FOV_RAD / 2
        if self.mode == "world":
            los = camera.theta + np.pi
            self.dynamic(
                key,
                lambda: self.axis.add_patch(Wedge(center=(x, y), r=CAM_DOF, theta1=math.degrees(los + beta2), theta2=math.degrees(los + beta1), **kwargs_region)),
                lambda patch: patch.set(center=(x, y), theta1=math.degrees(los + beta2), theta2=math.degrees(los + beta1)),
                visible=show_fov,
            )
        elif self.mode == "polar":
            # compute FOV boundary (in camera coordinates)
            # phi, boundary = np.concatenate([
            #   camera.camera_to_polar(np.linspace(1e-6, beta1), CAM_DOF),
            #   camera.camera_to_polar(beta1, np.linspace(CAM_DOF, 0)),
            #   camera.camera_to_polar(beta2, np.linspace(0, CAM_DOF)),
            #   camera.camera_to_polar(np.linspace(beta2, -1e-6), CAM_DOF),
            # ], axis=1)

            # compute FOV boundary (in polar coordinates almost analytically)
            phi1, boundary1 = camera.camera_to_polar(np.linspace(1e-6, beta1), CAM_DOF)
            phi2 = np.linspace(camera.camera_to_polar(beta1, CAM_DOF)[0], camera.theta)
            boundary2 = camera.ray_f(beta1)(phi2)
            phi3 = np.linspace(camera.theta, camera.camera_to_polar(beta2, CAM_DOF)[0])
            boundary3 = camera.ray_f(beta2)(phi3)
            phi4, boundary4 = camera.camera_to_polar(np.linspace(beta2, -1e-6), CAM_DOF)
            phi, boundary = np.concatenate([
                (phi1, boundary1),
                (phi2, boundary2),
                (phi3, boundary3),
                (phi4, boundary4),
            ], axis=1)

            # duplicate FOV for wrap-around
            if camera.theta < np.pi:
                phi_l, phi_r = phi, phi + 2*np.pi # add duplicate FOV to the right
            else:
                phi_l, phi_r = phi - 2*np.pi, phi # add duplicate FOV to the left
            # plot FOVs
            patches = []
            lines = []
            if CAM_DOF <= CAM_D:
                # plot FOVs separately
                patches = [
                    Polygon(np.array([phi_l, boundary]).T),
                    Polygon(np.array([phi_r, boundary]).T),
                ]
                lines = [phi_l, boundary, phi_r, boundary]
            else:
                # plot FOVs together
                phi, boundary = np.concatenate([
                    (phi_l, boundary),
                    (phi_r, boundary),
                ], axis=1)
                patches = [
                    Polygon(np.concatenate([
                        ([phi[0]], [0]),
                        (phi, boundary),
                        ([phi[-1]], [0])
                    ], axis=1).T),
                ]
                lines = [phi, boundary, [], []]
            self.dynamic_patch_collection(key + "_region", patches, **kwargs_region, visible=show_fov)
            self.dynamic_plot(key + "_boundary", *lines, **kwargs_boundary, visible=show_fov)
        
        # plot view circle
        key = "plot_camera:{}:view_circle".format(name)
        kwargs = dict(color=color, linestyle="--", linewidth=1, alpha=0.5)
        if self.mode == "world":
            self.static(key, lambda: self.axis.add_patch(Circle((0, 0), radius=CAM_D, **self.args_to_edgecolor(kwargs))), visible=show_view_circle)
        elif self.mode == "polar":
            self.static(key, lambda: self.axis.axhline(CAM_D, **kwargs), visible=show_view_circle)

    def plot_object(self, obj, show_object=True, show_points=False, show_pixels=False, show_bounds=True, n=200, name=None):
        color = self.colors["object"]

        # plot object surface
        key = "plot_object:{}:surface".format(name)
        kwargs = dict(color=color, linestyle="-", linewidth=1)
        phi = np.linspace(0, 2*np.pi, n)
        if self.mode == "world":
            x, y = polar_to_cartesian(phi, obj(phi))
            self.dynamic_plot(key, x, y, **kwargs, visible=show_object)
        elif self.mode == "polar":
            self.dynamic_plot(key, phi, obj(phi), **kwargs, visible=show_object)
        
        # plot surface points
        key = "plot_object:{}:points".format(name)
        kwargs = dict(**self.args_scatter(2), color=color, label="object surface points")
        if self.mode == "world":
            surface_points = polar_to_cartesian(obj.surface_points[0], obj.surface_points[1])
            self.dynamic_plot(key, surface_points[0], surface_points[1], **kwargs, visible=show_points)
        elif self.mode == "polar":
            self.dynamic_plot(key, obj.surface_points[0], obj.surface_points[1], **kwargs, visible=show_points)
        
        # highlight surface pixels
        key = "plot_object:{}:pixels".format(name)
        kwargs = dict(color=color, alpha=0.2)
        if self.mode == "world":
            patches = self.highlight_pixels(obj.surface_points)
            self.dynamic_patch_collection(key, patches, **self.args_to_facecolor(kwargs), visible=show_pixels)
        elif self.mode == "polar":
            if show_pixels: print("WARNING: not supported to plot pixels in polar plot")

        # plot object bounds
        key = "plot_object:{}:bounds".format(name)
        kwargs = dict(color=color, linestyle="--", linewidth=1, alpha=0.5)
        if self.mode == "world":
            patches = [Circle((0, 0), radius=OBJ_D_MIN), Circle((0, 0), radius=OBJ_D_MAX)]
            self.static(key, lambda: self.axis.add_collection(PatchCollection(patches, **self.args_to_edgecolor(kwargs))), visible=show_bounds)
        elif self.mode == "polar":
            self.static(key + "_lower", lambda: self.axis.axhline(OBJ_D_MIN, **kwargs), visible=show_bounds)
            self.static(key + "_upper", lambda: self.axis.axhline(OBJ_D_MAX, **kwargs), visible=show_bounds)

    def plot_confidence(self, gp, show_confidence=True, show_points=False, show_pixels=False, name=None):
        color = self.colors["confidence"]
        color_lower = self.colors["confidence_lower"]
        color_upper = self.colors["confidence_upper"]

        # plot GP mean
        key = "plot_confidence_bounds:{}:mean".format(name)
        kwargs = dict(color=color, linestyle="-", linewidth=1)
        if self.mode == "world":
            mean_x, mean_y = polar_to_cartesian(gp.x_eval, gp.mean)
            self.dynamic_plot(key, mean_x, mean_y, **kwargs, visible=show_confidence)
        elif self.mode == "polar":
            self.dynamic_plot(key, gp.x_eval, gp.mean, **kwargs, visible=show_confidence)
        
        # plot GP confidence bounds
        key = "plot_confidence_bounds:{}:bounds".format(name)
        kwargs = dict(color=color, alpha=0.2)
        if self.mode == "world":
            xy = gp.confidence_region().T
            self.dynamic(
                key,
                lambda: self.axis.add_patch(Polygon(xy, **self.args_to_facecolor(kwargs))),
                lambda patch: patch.set_xy(xy),
                visible=show_confidence,
            )
        elif self.mode == "polar":
            lower, upper = gp.confidence_boundary()
            def update_fill_between(poly_collection):
                fig, axis = plt.subplots() # create dummy figure
                poly_collection_new = axis.fill_between(lower[0], lower[1], upper[1])
                poly_collection.set_paths([path.vertices for path in poly_collection_new.get_paths()])
                plt.close(fig)
            self.dynamic(
                key,
                lambda: self.axis.fill_between(lower[0], lower[1], upper[1], **kwargs),
                update_fill_between,
                visible=show_confidence,
            )
        
        # plot discretization points of confidence bounds
        key = "plot_confidence_bounds:{}:points".format(name)
        kwargs_lower = dict(**self.args_scatter(2), color=color_lower)
        kwargs_upper = dict(**self.args_scatter(2), color=color_upper)
        data_lower = gp.lower_points
        data_upper = gp.upper_points
        if self.mode == "world":
            data_lower = polar_to_cartesian(data_lower[0], data_lower[1])
            data_upper = polar_to_cartesian(data_upper[0], data_upper[1])
            self.dynamic_plot(key + "_lower", data_lower[0], data_lower[1], **kwargs_lower, visible=show_points)
            self.dynamic_plot(key + "_upper", data_upper[0], data_upper[1], **kwargs_upper, visible=show_points)
        elif self.mode == "polar":
            self.dynamic_plot(key + "_lower", data_lower[0], data_lower[1], **kwargs_lower, visible=show_points)
            self.dynamic_plot(key + "_upper", data_upper[0], data_upper[1], **kwargs_upper, visible=show_points)
        
        # highlight discretization pixels of confidence bounds
        key = "plot_confidence_bounds:{}:pixels".format(name)
        kwargs_lower = dict(color=color_lower, alpha=0.2)
        kwargs_upper = dict(color=color_upper, alpha=0.2)
        data_lower = gp.lower_points
        data_upper = gp.upper_points
        if self.mode == "world":
            patches_lower = self.highlight_pixels(data_lower)
            patches_upper = self.highlight_pixels(data_upper)
            self.dynamic_patch_collection(key + "_lower", patches_lower, **self.args_to_facecolor(kwargs_lower), visible=show_pixels)
            self.dynamic_patch_collection(key + "_upper", patches_upper, **self.args_to_facecolor(kwargs_upper), visible=show_pixels)
        elif self.mode == "polar":
            if show_pixels: print("WARNING: not supported to plot pixels in polar plot")

    def plot_observations(self, observations, show=True, color=None, name=None):
        if color is None:
            color = "green"
        elif color in self.colors.keys():
            color = self.colors[color]

        # highlight all observed surface points
        key = "plot_observations:{}".format(name)
        kwargs = dict(**self.args_scatter(2), color=color)
        if self.mode == "world":
            self.plot_points(observations, highlight="pixelface", show=show, color=color, alpha=0.4, name=key)
            self.plot_points(observations, highlight="point", show=show, color=color, markersize=2, name=key)
        elif self.mode == "polar":
            self.plot_points(observations, highlight="point", show=show, color=color, markersize=2, name=key)

    def plot_points(self, points, highlight="point", show=True, name=None, **kwargs):
        if "color" not in kwargs.keys():
            kwargs["color"] = "orange"
        elif kwargs["color"] in self.colors.keys():
            kwargs["color"] = self.colors[kwargs["color"]]

        if highlight == "point":
            # highlight given points
            key = "plot_points:{}:point".format(name)
            style = {**self.args_scatter(4), **kwargs}
            if self.mode == "world":
                points = polar_to_cartesian(points[0], points[1])
                self.dynamic_plot(key, points[0], points[1], **style, visible=show)
            elif self.mode == "polar":
                self.dynamic_plot(key, points[0], points[1], **style, visible=show)
        elif highlight == "pixelface" or highlight == "pixeledge":
            # highlight pixels containing given points
            key = "plot_points:{}:pixel".format(name)
            style = {**dict(linestyle="-", linewidth=1), **kwargs}
            if highlight == "pixelface": 
                style = self.args_to_facecolor(style)
            elif highlight == "pixeledge":
                style = self.args_to_edgecolor(style)
            if self.mode == "world":
                patches = self.highlight_pixels(points)
                self.dynamic_patch_collection(key, patches, **style, visible=show)
            elif self.mode == "polar":
                if show: print("WARNING: not supported to plot pixels in polar plot")
