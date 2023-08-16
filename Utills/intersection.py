from shapely.geometry import Polygon, LineString


class TressPass:

    def __init__(self, person_coordinates, line_coordinates):

        self.person_coordinates = person_coordinates
        self.line_coordinates = line_coordinates

    def intersection(self):

        for xyxy, confidence, class_id, tracker_id in self.person_coordinates:

            p_x1, p_y1, p_x2, p_y2 = xyxy
            person_coord = [(p_x1, p_y1), (p_x1, p_y2), (p_x2, p_y1), (p_x2, p_y2)]

            rect_polygon = Polygon(person_coord)

            # Define the coordinates of the line's endpoints
            line_coords = list(self.line_coordinates)

            # Create a Shapely LineString object representing the line
            line = LineString(line_coords)

            # Check for intersection between the line and the rectangular object
            intersection = line.intersects(rect_polygon)
            if intersection:
                return True
            else:
                return False