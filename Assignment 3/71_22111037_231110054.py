import vtk
from vtk.util.numpy_support import vtk_to_numpy
import numpy as np
import sys

def load_vector_field(filename):
    reader = vtk.vtkXMLImageDataReader()
    reader.SetFileName(filename)
    reader.Update()
    return reader.GetOutput()

def rk4(x, h, vector_field):
    def get_vector(point):
        probe = vtk.vtkProbeFilter()
        probe.SetSourceData(vector_field)
        points = vtk.vtkPoints()
        points.InsertNextPoint(point)
        poly_data = vtk.vtkPolyData()
        poly_data.SetPoints(points)
        probe.SetInputData(poly_data)
        probe.Update()
        vectors = probe.GetOutput().GetPointData().GetVectors()
        if vectors is None:
            return np.array([0,0,0])  # return a zero vector if outside the data bounds
        return vtk_to_numpy(vectors).flatten()

    a = h * get_vector(x)
    b = h * get_vector(x + a / 2)
    c = h * get_vector(x + b / 2)
    d = h * get_vector(x + c)
    return x + (a + 2 * b + 2 * c + d) / 6


def trace_streamline(seed_point, vector_field, h=0.05, max_steps=1000):
    bounds = vector_field.GetBounds()
    points = [seed_point]
    # Trace forward
    point = seed_point[:]
    for _ in range(max_steps):
        next_point = rk4(point, h, vector_field)
        if not (bounds[0] <= next_point[0] <= bounds[1] and bounds[2] <= next_point[1] <= bounds[3] and bounds[4] <= next_point[2] <= bounds[5]):
            break
        points.append(next_point)
        point = next_point
    
    # Trace backward
    point = seed_point[:]
    for _ in range(max_steps):
        next_point = rk4(point, -h, vector_field)
        if not (bounds[0] <= next_point[0] <= bounds[1] and bounds[2] <= next_point[1] <= bounds[3] and bounds[4] <= next_point[2] <= bounds[5]):
            break
        points.insert(0, next_point)
        point = next_point

    return points

def save_streamline(streamline_points, filename):
    points = vtk.vtkPoints()
    lines = vtk.vtkCellArray()

    for point_id, point in enumerate(streamline_points):
        points.InsertNextPoint(point)
        if point_id > 0:
            line = vtk.vtkLine()
            line.GetPointIds().SetId(0, point_id - 1)
            line.GetPointIds().SetId(1, point_id)
            lines.InsertNextCell(line)

    poly_data = vtk.vtkPolyData()
    poly_data.SetPoints(points)
    poly_data.SetLines(lines)

    writer = vtk.vtkXMLPolyDataWriter()
    writer.SetFileName(filename)
    writer.SetInputData(poly_data)
    writer.Write()

def main():
    
    filename = sys.argv[1]
    seed_point = [float(sys.argv[2]), float(sys.argv[3]), float(sys.argv[4])]
    vector_field = load_vector_field(filename)
    streamline = trace_streamline(seed_point, vector_field)

    save_streamline(streamline, "output_streamline.vtp")

if __name__ == "__main__":
    main()
