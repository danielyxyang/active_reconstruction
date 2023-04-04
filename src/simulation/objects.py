import numpy as np

import parameters as params
from utils.math import polar_to_cartesian, polar_to_pixel
from utils.tools import Profiler, LoopChecker


class Object():
    name = "object"

    def __init__(self, obj_f, args={}, discretize=True):
        """Construct object for given object function.
        
        Note: `obj_f` can assume that a np.array of type float containing
        polar angles in the range [0,2pi) is passed as an argument.

        Args:
            obj_f: function returning radial distances for given polar angles
            args: keyword arguments used to build this object
        """
        self.profiler = Profiler()

        self.obj_f = obj_f
        self.obj_name = self.name
        self.args = args
        self.surface_points = np.empty((2, 0))
        if discretize:
            self._discretize()
    
    @classmethod
    def build(cls, *args, **kwargs):
        """Build object with function instead of constructor call.
        
        Note: This function is used to interactively construct objects using ipywidgets."""
        return cls(*args, **kwargs)

    def __str__(self):
        return self.obj_name + "".join(["_{}{}".format(name[0], arg if arg % 1 else int(arg)) for name, arg in self.args.items()])
    
    def __repr__(self):
        return str(self)
    
    def __eq__(self, other):
        if isinstance(other, Object):
            return str(self) == str(other)
        return False
    
    def __hash__(self):
        return hash(str(self))

    def __call__(self, phi):
        """Return polar coordinates of object surface at phi."""
        phi = np.asarray(phi, dtype=float) % (2*np.pi) # sanitize phi
        return self.obj_f(phi)
    
    def cc(self, phi):
        """Return cartesian coordinates of object surface at phi."""
        return polar_to_cartesian(phi, self(phi))
    
    def pc(self, phi):
        """Return pixel coordinates of object surface at phi."""
        return polar_to_pixel(phi, self(phi))

    def to_json(self):
        return {
            "object": self.obj_name,
            "args": self.args,
        }

    # PRIVATE METHODS

    def _discretize(self):
        """Compute polar coordinates of points on object surface."""
        # initialize profiling
        n_iters = []

        with self.profiler.cm("discretization (object)"):
            # initialize list of surface phis
            surface_phis = []
            phi = 0
            CHECKER = LoopChecker("Object:discretize")
            while phi < 2*np.pi:
                CHECKER()
                # add surface point to list
                surface_phis.append(phi)
                
                # compute phi of surface point (approximately) centered in next pixel
                phi_next1, n_iter1 = self.__find_phi(phi, d_phi=params.GRID_H, target_mode="4-neighbors")
                if phi_next1 is None:
                    print("WARNING: failed to find left phi for discretization of object")
                    break
                phi_next2, n_iter2 = self.__find_phi(phi_next1, d_phi=params.GRID_H, target_mode="this")
                if phi_next2 is None:
                    print("WARNING: failed to find right phi for discretization of object")
                    break
                phi = (phi_next1 + phi_next2) / 2
                n_iters += [n_iter1, n_iter2]
            self.surface_points = np.array([surface_phis, self(surface_phis)])
        
        # add number of iterations to profiler
        if len(n_iters) > 0:
            self.profiler.set_info("discretization (object)", "{:3.1f} iterations".format(np.mean(n_iters)))

    def __find_phi(self, phi, d_phi=1, target_mode="this", alpha=0.1, max_iter=25):
        """TODO"""

        # initialize target pixels
        px, py = self.pc(phi)
        if target_mode == "this":
            target_pixels = [(px, py)]
        elif target_mode == "4-neighbors":
            target_pixels = [
                (px+1, py),
                (px, py+1),
                (px-1, py),
                (px, py-1),
            ]
        elif target_mode == "8-neighbors":
            target_pixels = [
                (px+1, py), (px+1, py+1),
                (px, py+1), (px-1, py+1),
                (px-1, py), (px-1, py-1),
                (px, py-1), (px+1, py-1),
            ]
        else:
            raise ValueError("invalid argument target_mode")

        # initialize search boundaries
        phi_left = phi
        phi_right = phi + d_phi
        CHECKER = LoopChecker("Object:__find_phi")
        while tuple(self.pc(phi_right)) == tuple(self.pc(phi_left)): # ensure search boundaries mapped to different pixels
            CHECKER()
            phi_right += d_phi
        
        # binary search for intersection of surface with pixel edge
        phi_found = None
        n_iter = 0
        # iterate as long as no phi has been found or is not exact enough (but at most max_iter times)
        while (phi_found is None or np.linalg.norm(self.cc(phi_right) - self.cc(phi_left)) > params.GRID_H * alpha) and n_iter < max_iter:
            phi_middle = (phi_left + phi_right) / 2
            pixel_middle = tuple(self.pc(phi_middle))
            
            # store potential next phi
            if pixel_middle in target_pixels:
                phi_found = phi_middle
            
            # update search boundaries
            if pixel_middle == tuple(self.pc(phi_left)):
                phi_left = phi_middle
            else:
                phi_right = phi_middle
            
            n_iter += 1

        return phi_found, n_iter
    
    # STATIC METHODS

    @staticmethod
    def from_json(obj_json, **kwargs):
        for object in OBJECTS:
            if obj_json["object"] == object.name:
                return object(**obj_json["args"], **kwargs)
        print("WARNING: not able to find object \"{}\"".format(obj_json["object"]))
        return None
    
    @staticmethod
    def construct_line(p1, p2):
        """Return polar function parameterizing line going through the given points.
        
        Args:
            p1: polar coordinate of first point
            p2: polar coordinate of second point
        """
        # sort points with increasing polar angle
        if p1[0] > p2[0]:
            p1, p2 = p2, p1
        
        # compute cartesian coordinates of endpoints
        p1 = polar_to_cartesian(p1[0], p1[1])
        p2 = polar_to_cartesian(p2[0], p2[1])
        
        # check for numerical stability of division
        if p2[0] - p1[0] > p2[1] - p1[1]:
            # compute parameters of line equation in x-coordinate
            m = (p2[1] - p1[1]) / (p2[0] - p1[0])
            b = p1[1] - m * p1[0]
            # transform line equation into polar coordinates
            f = lambda phi: b / (np.sin(phi) - m * np.cos(phi))
        else:
            # compute parameters of line equation in y-coordinate
            m = (p2[0] - p1[0]) / (p2[1] - p1[1])
            b = p1[0] - m * p1[1]
            # transform line equation into polar coordinates
            f = lambda phi: b / (np.cos(phi) - m * np.sin(phi))

        return f


class EllipseObject(Object):
    name = "ellipse"

    def __init__(self, a=None, b=None, **kwargs):
        """Return object with the shape of an ellipse using a linear function.
        
        Args:
            a: semi-major axis aligned with x-axis (defaults to average of object bounds)
            b: semi-minor axis aligned with y-axis (defaults to average of object bounds)
        """
        a = params.OBJ_D_AVG if a is None else a
        a = np.clip(a, params.OBJ_D_MIN, params.OBJ_D_MAX)
        b = params.OBJ_D_AVG if b is None else b
        b = np.clip(b, params.OBJ_D_MIN, params.OBJ_D_MAX)
        args = dict(a=a, b=b)
        def obj_f(phi):
            return a * b / np.sqrt((b * np.cos(phi)) ** 2 + (a * np.sin(phi)) ** 2)
        super().__init__(obj_f, args=args, **kwargs)
    

class SquareObject(Object):
    name = "square"

    def __init__(self, width=None, **kwargs):
        """Return object with the shape of a square using a piecewise function.
        
        Args:
            width: width (defaults to twice the average of object bounds)
        """
        width = 2*params.OBJ_D_AVG if width is None else width
        width = np.clip(width, 2 * params.OBJ_D_MIN, 2 * params.OBJ_D_MAX / np.sqrt(2))
        args = dict(width=width)
        def obj_f(phi):
            phi = (phi - np.pi/4) % (np.pi/2) + np.pi/4 # map phi to [1/4*pi, 3/4*pi)
            return width/2 / np.sin(phi)
        super().__init__(obj_f, args=args, **kwargs)


class FlowerObject(Object):
    name = "flower"

    def __init__(self, amplitude=None, frequency=5, **kwargs):
        """Return object with the shape of a flower using a cosine function.
    
        Args:
            amplitude: amplitude (defaults to halfway between object bounds)
            frequency: integer-valued frequency
        """
        amplitude = (params.OBJ_D_MAX - params.OBJ_D_MIN) / 2 if amplitude is None else amplitude
        amplitude = np.clip(amplitude, 0, (params.OBJ_D_MAX - params.OBJ_D_MIN) / 2)
        frequency = int(frequency)
        args=dict(frequency=frequency, amplitude=amplitude)
        def obj_f(phi):
            return amplitude * np.cos(frequency * phi) + params.OBJ_D_AVG
        super().__init__(obj_f, args=args, **kwargs)


class PolygonObject(Object):
    name = "polygon"
    polygons = None
    polygon_names = ["diamond", "convex", "concave", "star"] # names of predefined polygons

    def __init__(self, vertices, name=None, **kwargs):
        """Return object with the shape of a polygon.
        
        Args:
            vertices: list of polar coordinates of the vertices
            name: name used to uniquely identify polygon 
        """
        args = dict(vertices=vertices, name=name)

        # TODO add sanity checks (e.g. vertices sorted by phi, no duplicates and also no 0 and 2pi)
        # extend list of vertices with wrap-around to cover [0,...] and [...,2pi]
        vertices = np.concatenate((
            [(vertices[-1, 0] - 2*np.pi, vertices[-1, 1])],
            vertices,
            [(vertices[0, 0] + 2*np.pi, vertices[0, 1])],
        ))
        # build piecewise functions for each polygon edge
        line_fs = [(p1[0], p2[0], Object.construct_line(p1, p2)) for p1, p2 in zip(vertices[:-1], vertices[1:])]
        def obj_f(phi):
            result = np.zeros_like(phi, dtype=float)
            for phi1, phi2, line_f in line_fs:
                mask = np.logical_and(phi1 <= phi, phi <= phi2)
                result[mask] = line_f(phi[mask])
            return result
        super().__init__(obj_f, args=args, **kwargs)
    
    def __str__(self):
        return "{}_{}".format(self.obj_name, self.args["name"])

    @staticmethod
    def build(name):
        """Build pre-defined polygons.
        
        Note: This function overwrites the Object.build method and is used to
        interactively select polygons using ipywidgets."""
        t4 = np.linspace(0, 2*np.pi, num=4, endpoint=False)
        t6 = np.linspace(0, 2*np.pi, num=6, endpoint=False)
        t10 = np.linspace(0, 2*np.pi, num=10, endpoint=False)
        if   name == "diamond": return PolygonObject(np.array([t4, [3,4,6,4]]).T, name=name)
        elif name == "convex":  return PolygonObject(np.array([t6, [4,8,6,8,4,2]]).T, name=name)
        elif name == "concave": return PolygonObject(np.array([t10, [5,8,4,3,8,7,7,5,3,7]]).T, name=name)
        elif name == "star":    return PolygonObject(np.array([t10, [5,8,4,6,2,7,4,7,2,8]]).T, name=name)

OBJECTS = [EllipseObject, SquareObject, FlowerObject, PolygonObject]
