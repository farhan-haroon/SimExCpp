#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped, Point, PoseWithCovarianceStamped
from nav2_msgs.action import NavigateToPose
from rclpy.action import ActionClient
from rclpy.callback_groups import ReentrantCallbackGroup
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import ColorRGBA
from nav_msgs.msg import OccupancyGrid, MapMetaData
import math
import numpy as np
from rclpy.qos import QoSProfile, QoSDurabilityPolicy, QoSReliabilityPolicy, QoSHistoryPolicy
from sensor_msgs.msg import PointCloud2
from collections import defaultdict, deque
import struct

# -------------------------
# Union-Find (Disjoint Set) for Kruskal
# -------------------------
class DisjointSet:
    def __init__(self, n):
        self.parent = list(range(n))

    def find(self, u):
        if self.parent[u] != u:
            self.parent[u] = self.find(self.parent[u])
        return self.parent[u]

    def union(self, u, v):
        pu, pv = self.find(u), self.find(v)
        if pu != pv:
            self.parent[pu] = pv
            return True
        return False

# -------------------------
# Kruskal MST STC Node
# -------------------------
class KruskalSTCNode(Node):
    def __init__(self):
        super().__init__('kruskal_stc_node')

        # -------------------------
        # Robot & Grid Parameters (will be set from map)
        # -------------------------
        robot_length = 0.80  # meters
        robot_width  = 0.80  # meters
        
        # Initialize parameters
        self.map_data = None
        self.map_info = None
        self.map_width = 0
        self.map_height = 0
        self.map_resolution = 0.0
        self.map_origin_x = 0.0
        self.map_origin_y = 0.0
        
        self.subcell_size = max(robot_length, robot_width) * 1.2  # slightly larger than robot
        self.subcell_per_cell = 2
        self.cell_size = self.subcell_size * self.subcell_per_cell
        
        # These will be calculated from map
        self.main_grid_size_x = 0
        self.main_grid_size_y = 0
        self.total_grid_x = 0
        self.total_grid_y = 0
        
        self.origin_x = 0.0
        self.origin_y = 0.0
        self.safe_distance = float(self.subcell_size)
        self.obstacle_detected = False
        
        # New: Occupancy/status grid for major cells ('free','obstacle','invalid')
        self.major_cell_occupancy = None

        # Reachability mask over the raw occupancy grid (2D numpy bool)
        self.reachable_mask = None
        
        # Robot pose for BFS seed
        self.robot_pose_received = False
        self.robot_pose_x = 0.0
        self.robot_pose_y = 0.0
        
        # -------------------------
        # Subscribers, Publishers & Nav2 client
        # -------------------------
        # QoS for /map (match typical map_server QoS)
        map_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=10
        )

        self.map_sub = self.create_subscription(
            OccupancyGrid,
            '/map',
            self.map_callback,
            map_qos
        )
        
        # Subscribe to amcl_pose for robot initial pose (map frame)
        amcl_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1
        )

        self.pose_sub = self.create_subscription(
            PoseWithCovarianceStamped,
            '/amcl_pose',
            self.pose_callback,
            amcl_qos
        )

        marker_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1
            )
        
        self.marker_pub = self.create_publisher(MarkerArray, 'stc_markers', marker_qos)
        self._cb_group = ReentrantCallbackGroup()
        self.nav_client = ActionClient(self, NavigateToPose, 'navigate_to_pose', callback_group=self._cb_group)
        
        # -------------------------
        # Wait for map and robot pose before proceeding
        # -------------------------
        self.get_logger().info("Waiting for map data and robot pose...")
        self.map_received = False
        
        # Timer to check if map & pose are ready and start planning
        self.timer = self.create_timer(1.0, self.check_map_and_plan)

        # LiDAR subscription (optional)
        lidar_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            durability=QoSDurabilityPolicy.VOLATILE,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=10
        )
        self.lidar_sub = self.create_subscription(PointCloud2, '/velodyne_points', self.lidar_callback, lidar_qos)

    def pose_callback(self, msg: PoseWithCovarianceStamped):
        # store robot pose (map frame)
        p = msg.pose.pose.position
        self.robot_pose_x = float(p.x)
        self.robot_pose_y = float(p.y)
        self.robot_pose_received = True
        
        self.get_logger().info(
            f"AMCL pose received: ({self.robot_pose_x:.2f}, "
            f"{self.robot_pose_y:.2f})"
        )

    def lidar_callback(self, msg):
        # simple near-obstacle flag (keeps previous behavior)
        pts = []
        for i in range(0, len(msg.data), msg.point_step):
            try:
                x, y, z = struct.unpack_from('fff', msg.data, offset=i)
                if not (np.isnan(x) or np.isnan(y) or np.isnan(z)):
                    if abs(z) < 1.0:
                        pts.append((float(x), float(y), float(z)))
            except struct.error:
                break
        if not pts:
            self.obstacle_detected = False
            return
        arr = np.array(pts)
        dists = np.linalg.norm(arr[:, 0:2], axis=1)
        self.obstacle_detected = int(np.count_nonzero(dists < float(self.safe_distance))) > 0

    def map_callback(self, msg):
        """Callback to receive map data"""
        # Save raw map and metadata
        self.map_data = list(msg.data)
        self.map_info = msg.info
        self.map_width = msg.info.width
        self.map_height = msg.info.height
        self.map_resolution = msg.info.resolution
        self.map_origin_x = msg.info.origin.position.x
        self.map_origin_y = msg.info.origin.position.y
        
        self.get_logger().info(f"Map received: {self.map_width}x{self.map_height} cells, "
                              f"resolution: {self.map_resolution:.3f}m/cell")
        self.get_logger().info(f"Map origin: ({self.map_origin_x:.2f}, {self.map_origin_y:.2f})")
        
        self.map_received = True

    def check_map_and_plan(self):
        """Check if map and robot pose are ready and start planning"""
        if self.map_received and self.robot_pose_received:
            self.timer.cancel()
            # Build reachable mask from robot pose
            self.build_reachable_mask_from_robot()
            self.initialize_planning()
            self.create_stc_path()
            self.publish_markers()
            self.send_next_pose(0)
        else:
            self.get_logger().info("Still waiting for map or robot pose...")

    def build_reachable_mask_from_robot(self):
        """Build a boolean reachable_mask (shape: map_height x map_width)
        BFS from robot's map cell through free cells (value 0..50).
        Unknown (-1) and occupied (>50) are not traversable.
        """
        if self.map_data is None:
            return
        # Convert map_data to 2D numpy array (row-major)
        arr = np.array(self.map_data, dtype=int).reshape((self.map_height, self.map_width))
        free_mask = (arr >= 0) & (arr <= 50)  # free cells
        reachable = np.zeros_like(free_mask, dtype=bool)

        # Compute robot cell indices
        col = int((self.robot_pose_x - self.map_origin_x) / self.map_resolution)
        row = int((self.robot_pose_y - self.map_origin_y) / self.map_resolution)

        # clamp / fallback to first free cell if robot pose out-of-bounds or not free
        if not (0 <= row < self.map_height and 0 <= col < self.map_width) or not free_mask[row, col]:
            # try to find nearest free cell by scanning small radius then global fallback
            found = False
            max_r = max(self.map_height, self.map_width)
            for radius in range(0, max_r):
                rmin = max(0, row - radius)
                rmax = min(self.map_height - 1, row + radius)
                cmin = max(0, col - radius)
                cmax = min(self.map_width - 1, col + radius)
                sub = free_mask[rmin:rmax+1, cmin:cmax+1]
                if sub.any():
                    # pick first free cell in subwindow
                    idx = np.argwhere(sub)[0]
                    rr = rmin + idx[0]
                    cc = cmin + idx[1]
                    row, col = int(rr), int(cc)
                    found = True
                    break
            if not found:
                # global fallback: pick first free cell in entire map
                idxs = np.argwhere(free_mask)
                if idxs.size == 0:
                    self.get_logger().error("No free cells in map to start BFS!")
                    self.reachable_mask = reachable
                    return
                row, col = int(idxs[0][0]), int(idxs[0][1])

        # BFS queue
        q = deque()
        if free_mask[row, col]:
            reachable[row, col] = True
            q.append((row, col))
        else:
            self.reachable_mask = reachable
            return

        # 4-neighbor BFS
        while q:
            r, c = q.popleft()
            for dr, dc in ((1,0),(-1,0),(0,1),(0,-1)):
                rr = r + dr
                cc = c + dc
                if 0 <= rr < self.map_height and 0 <= cc < self.map_width:
                    if (not reachable[rr, cc]) and free_mask[rr, cc]:
                        reachable[rr, cc] = True
                        q.append((rr, cc))

        self.reachable_mask = reachable
        self.get_logger().info("Built reachable mask from robot pose.")

    def initialize_planning(self):
        """Initialize planning parameters based on map"""
        # Calculate grid dimensions based on map size
        map_width_meters = self.map_width * self.map_resolution
        map_height_meters = self.map_height * self.map_resolution
        
        # Calculate number of major cells that fit in the map
        self.main_grid_size_x = int(map_width_meters / self.cell_size)
        self.main_grid_size_y = int(map_height_meters / self.cell_size)
        
        # Adjust cell size if map doesn't fit perfectly
        if self.main_grid_size_x == 0:
            self.main_grid_size_x = 1
        if self.main_grid_size_y == 0:
            self.main_grid_size_y = 1
            
        self.cell_size = min(map_width_meters / self.main_grid_size_x, 
                           map_height_meters / self.main_grid_size_y)
        self.subcell_size = self.cell_size / self.subcell_per_cell
        
        self.total_grid_x = self.main_grid_size_x * self.subcell_per_cell
        self.total_grid_y = self.main_grid_size_y * self.subcell_per_cell
        
        # Set origin to map origin
        self.origin_x = self.map_origin_x
        self.origin_y = self.map_origin_y
        
        # Initialize occupancy grid for major cells with 'free' as default
        self.major_cell_occupancy = [['free' for _ in range(self.main_grid_size_x)] 
                                    for _ in range(self.main_grid_size_y)]
        
        self.get_logger().info(f"Planning grid: {self.main_grid_size_x}x{self.main_grid_size_y} major cells")
        self.get_logger().info(f"Subgrid: {self.total_grid_x}x{self.total_grid_y} subcells")
        self.get_logger().info(f"Cell size: {self.cell_size:.2f}m, Subcell size: {self.subcell_size:.2f}m")
        self.get_logger().info(f"Origin: ({self.origin_x:.2f}, {self.origin_y:.2f})")

    # -------------------------
    # Helper: classify a subcell
    # returns 'free', 'obstacle', or 'invalid' (out-of-bounds or unknown/outside-wall or unreachable)
    # -------------------------
    def subcell_status(self, r, c):
        """Return 'free', 'obstacle' or 'invalid' for a subcell index (r,c)."""
        if self.map_data is None:
            return 'free'
        # bounding box of subcell (world coordinates)
        minx = self.origin_x + c * self.subcell_size
        maxx = minx + self.subcell_size
        miny = self.origin_y + r * self.subcell_size
        maxy = miny + self.subcell_size

        # convert bounding box to map grid indices
        col_min = int(math.floor((minx - self.map_origin_x) / self.map_resolution))
        col_max = int(math.ceil((maxx - self.map_origin_x) / self.map_resolution)) - 1
        row_min = int(math.floor((miny - self.map_origin_y) / self.map_resolution))
        row_max = int(math.ceil((maxy - self.map_origin_y) / self.map_resolution)) - 1

        # if any overlap extends outside map -> invalid
        if col_min < 0 or row_min < 0 or col_max >= self.map_width or row_max >= self.map_height:
            return 'invalid'

        # scan all overlapping map cells
        for rr in range(row_min, row_max + 1):
            for cc in range(col_min, col_max + 1):
                idx = rr * self.map_width + cc
                if idx < 0 or idx >= len(self.map_data):
                    return 'invalid'
                val = self.map_data[idx]
                # unknown/outside wall
                if val is None or val < 0:
                    return 'invalid'
                # obstacle
                if val > 50:
                    return 'obstacle'
                # unreachable (not in reachable_mask)
                if self.reachable_mask is None or not self.reachable_mask[rr, cc]:
                    return 'invalid'
        return 'free'

    # Keep old boolean API but now only True for obstacles
    def is_subcell_occupied(self, r, c):
        status = self.subcell_status(r, c)
        return status == 'obstacle'

    # -------------------------
    # Determine major cell status: 'free', 'obstacle', or 'invalid'
    # A major cell is 'invalid' if any overlapping map cell is unknown/outside or unreachable.
    # 'obstacle' if any overlapping map cell > 50.
    # otherwise 'free'.
    # -------------------------
    def get_major_cell_status(self, r, c):
        if self.map_data is None:
            return 'free'
        # Define cell boundaries in world coordinates
        cell_min_x = self.origin_x + c * self.cell_size
        cell_max_x = cell_min_x + self.cell_size
        cell_min_y = self.origin_y + r * self.cell_size
        cell_max_y = cell_min_y + self.cell_size

        # convert bounding box to map grid indices
        col_min = int(math.floor((cell_min_x - self.map_origin_x) / self.map_resolution))
        col_max = int(math.ceil((cell_max_x - self.map_origin_x) / self.map_resolution)) - 1
        row_min = int(math.floor((cell_min_y - self.map_origin_y) / self.map_resolution))
        row_max = int(math.ceil((cell_max_y - self.map_origin_y) / self.map_resolution)) - 1

        # if any overlap extends outside map -> invalid
        if col_min < 0 or row_min < 0 or col_max >= self.map_width or row_max >= self.map_height:
            return 'invalid'

        # scan all overlapping map cells
        for rr in range(row_min, row_max + 1):
            for cc in range(col_min, col_max + 1):
                idx = rr * self.map_width + cc
                if idx < 0 or idx >= len(self.map_data):
                    return 'invalid'
                val = self.map_data[idx]
                if val is None or val < 0:
                    return 'invalid'
                if val > 50:
                    return 'obstacle'
                # unreachable check
                if self.reachable_mask is None or not self.reachable_mask[rr, cc]:
                    return 'invalid'

        # Also check subcells inside major cell (conservative)
        subcell_start_r = r * self.subcell_per_cell
        subcell_start_c = c * self.subcell_per_cell
        for sr in range(self.subcell_per_cell):
            for sc in range(self.subcell_per_cell):
                s_status = self.subcell_status(subcell_start_r + sr, subcell_start_c + sc)
                if s_status == 'invalid':
                    return 'invalid'
                if s_status == 'obstacle':
                    return 'obstacle'

        return 'free'

    def create_stc_path(self):
        """Create the STC path based on map using spanning tree edges"""
        # -------------------------
        # Major nodes = center of each major cell (in meters)
        # Only create nodes for non-occupied cells
        # -------------------------
        self.major_nodes = []
        self.major_node_indices = {}  # Maps (r,c) to node index
        self.valid_major_cells = []  # List of (r,c) for valid cells
        self.major_to_subcell = {}  # Map from major cell to its subcells
        
        node_idx = 0
        for r in range(self.main_grid_size_y):
            for c in range(self.main_grid_size_x):
                # Check status of this major cell: 'free','obstacle','invalid'
                status = self.get_major_cell_status(r, c)
                self.major_cell_occupancy[r][c] = status
                
                if status == 'free':
                    center_x = self.origin_x + (c + 0.5) * self.cell_size
                    center_y = self.origin_y + (r + 0.5) * self.cell_size
                    self.major_nodes.append((center_y, center_x))  # row, col in meters for MST
                    self.major_node_indices[(r, c)] = node_idx
                    self.valid_major_cells.append((r, c))
                    
                    # Store subcells for this major cell
                    subcells = []
                    for sr in range(self.subcell_per_cell):
                        for sc in range(self.subcell_per_cell):
                            subcell_r = r * self.subcell_per_cell + sr
                            subcell_c = c * self.subcell_per_cell + sc
                            if self.subcell_status(subcell_r, subcell_c) == 'free':
                                subcells.append((subcell_r, subcell_c))
                    self.major_to_subcell[(r, c)] = subcells
                    
                    node_idx += 1
        
        if not self.major_nodes:
            self.get_logger().error("No valid major cells found! All cells are occupied or out of bounds.")
            return
        
        self.get_logger().info(f"Found {len(self.major_nodes)} valid major cells out of {self.main_grid_size_x * self.main_grid_size_y} total cells")
        
        # -------------------------
        # MST on major nodes (only valid cells)
        # -------------------------
        self.major_edges = self.generate_major_edges()
        self.major_mst_edges = self.kruskal_mst_major()
        
        # -------------------------
        # Build adjacency list for MST
        # -------------------------
        self.mst_adjacency = defaultdict(list)
        for u, v in self.major_mst_edges:
            self.mst_adjacency[u].append(v)
            self.mst_adjacency[v].append(u)
        
        # -------------------------
        # Find MST components for traversal
        # -------------------------
        visited_mst = set()
        self.mst_components = []
        
        for node_idx in range(len(self.major_nodes)):
            if node_idx not in visited_mst:
                # BFS to find component
                component = []
                stack = [node_idx]
                while stack:
                    current = stack.pop()
                    if current in visited_mst:
                        continue
                    visited_mst.add(current)
                    component.append(current)
                    for neighbor in self.mst_adjacency[current]:
                        if neighbor not in visited_mst:
                            stack.append(neighbor)
                self.mst_components.append(component)
        
        self.get_logger().info(f"MST has {len(self.mst_components)} components")
        
        # -------------------------
        # Compute robot subcell (if pose known) so we can start coverage there
        # -------------------------
        robot_subcell = None
        if self.robot_pose_received:
            try:
                rs = int((self.robot_pose_y - self.origin_y) / self.subcell_size)
                cs = int((self.robot_pose_x - self.origin_x) / self.subcell_size)
                if 0 <= rs < self.total_grid_y and 0 <= cs < self.total_grid_x:
                    # If that exact subcell isn't free, find a nearby free subcell (small radius)
                    if self.subcell_status(rs, cs) == 'free':
                        robot_subcell = (rs, cs)
                    else:
                        found = False
                        # search a small radius first to avoid jumping far
                        for rad in range(1, 6):
                            for dr in range(-rad, rad+1):
                                for dc in range(-rad, rad+1):
                                    nr = rs + dr
                                    nc = cs + dc
                                    if 0 <= nr < self.total_grid_y and 0 <= nc < self.total_grid_x:
                                        if self.subcell_status(nr, nc) == 'free':
                                            robot_subcell = (nr, nc)
                                            found = True
                                            break
                                if found:
                                    break
                            if found:
                                break
                        # if still not found, leave robot_subcell as None
            except Exception:
                robot_subcell = None

        # -------------------------
        # Generate STC path for each component
        # -------------------------
        self.path = []
        self.component_paths = []          # <-- added: keep per-component paths for visualization
        for component in self.mst_components:
            if not component:
                continue
                
            # Find the major cell coordinates for each node in component
            node_to_cell = {}
            cell_to_node = {}
            for node_idx in component:
                for (r, c), idx in self.major_node_indices.items():
                    if idx == node_idx:
                        node_to_cell[node_idx] = (r, c)
                        cell_to_node[(r, c)] = node_idx
                        break
            
            if not node_to_cell:
                continue
                
            # Start from first node in component
            start_node = component[0]
            start_cell = node_to_cell[start_node]
            
            # Generate STC path for this component
            component_path = self.generate_component_stc_path(start_cell, cell_to_node)

            # --- NEW: rotate component_path to start from robot_subcell (if it lies in this component) ---
            # --- Make robot start moving LEFT from its initial position ---
            if robot_subcell and component_path:
                # If robot subcell itself is in the path, rotate to start there
                if robot_subcell in component_path:
                    idx0 = component_path.index(robot_subcell)
                    component_path = component_path[idx0:] + component_path[:idx0]
        
                    # Now check the NEXT move direction from robot's position
                    if len(component_path) > 1:
                        next_r, next_c = component_path[1]
                        robot_r, robot_c = component_path[0]
            
                        # Calculate direction vector
                        dr = next_r - robot_r
                        dc = next_c - robot_c
                        # If next move is NOT left (0, -1), we need to find a different starting point
                        # that WILL make the robot move left first
                        if dc != -1:  # Not moving left
                            # Look ahead in the path for a point where left movement occurs
                            for i in range(1, min(20, len(component_path))):  # Check next 20 points
                                if i+1 < len(component_path):
                                    curr_r, curr_c = component_path[i]
                                    next_r2, next_c2 = component_path[i+1]
                                    if next_c2 - curr_c == -1:  # This is a left movement
                                        # Rotate to start from this point
                                        component_path = component_path[i:] + component_path[:i]
                                        break
                
                            # If we couldn't find a left movement, force starting from leftmost point
                            # in the current major cell
                            robot_major = (robot_subcell[0] // self.subcell_per_cell, 
                                           robot_subcell[1] // self.subcell_per_cell)
                
                            # Find all subcells in robot's major cell that are in the path
                            robot_major_subcells = [(sr, sc) for (sr, sc) in component_path 
                                                   if (sr // self.subcell_per_cell, sc // self.subcell_per_cell) == robot_major]
                
                            if robot_major_subcells:
                                # Choose the LEFTmost subcell (smallest column index)
                                leftmost_subcell = min(robot_major_subcells, key=lambda x: x[1])
                                if leftmost_subcell in component_path:
                                    new_idx = component_path.index(leftmost_subcell)
                                    component_path = component_path[new_idx:] + component_path[:new_idx]
            else:
                 # If robot's major cell belongs to this component, pick leftmost node in component_path
                robot_major = (robot_subcell[0] // self.subcell_per_cell, 
                                robot_subcell[1] // self.subcell_per_cell)
                if robot_major in cell_to_node:
                    # Find all subcells in robot's major cell that are in the path
                    robot_major_subcells = [(sr, sc) for (sr, sc) in component_path 
                                            if (sr // self.subcell_per_cell, sc // self.subcell_per_cell) == robot_major]
                    if robot_major_subcells:
                        # Start from LEFTmost subcell in robot's major cell
                        leftmost_subcell = min(robot_major_subcells, key=lambda x: x[1])
                        if leftmost_subcell in component_path:
                            new_idx = component_path.index(leftmost_subcell)
                            component_path = component_path[new_idx:] + component_path[:new_idx]












            # Save per-component path for plotting and also append to the global path used for navigation
            self.component_paths.append(component_path)
            self.path.extend(component_path)
        
        self.get_logger().info(f"Generated STC path with {len(self.path)} waypoints")

    def generate_component_stc_path(self, start_cell, cell_to_node):
        """Generate STC path for a connected MST component"""
        path = []
        visited_subcells = set()
        
        # Start from a corner subcell of the starting major cell
        start_r, start_c = start_cell
        # Prefer the leftmost available subcell inside the starting major cell so traversal favors left first
        start_subcell_r = start_r * self.subcell_per_cell
        start_subcell_c = start_c * self.subcell_per_cell
        try:
            subcells_in_major = self.major_to_subcell.get((start_r, start_c), [])
            if subcells_in_major:
                # choose subcell with smallest column (leftmost), tie-breaker: smallest row (top)
                subcells_sorted = sorted(subcells_in_major, key=lambda x: (x[1], x[0]))
                start_subcell_r, start_subcell_c = subcells_sorted[0]
        except Exception:
            # fallback to original corner if anything goes wrong
            start_subcell_r = start_r * self.subcell_per_cell
            start_subcell_c = start_c * self.subcell_per_cell

        # We'll traverse in DFS order along MST edges
        stack = [(start_subcell_r, start_subcell_c, -1, -1)]  # (r, c, prev_r, prev_c)
        
        while stack:
            r, c, pr, pc = stack.pop()
            
            # Skip if this subcell is not free
            if self.subcell_status(r, c) != 'free':
                continue
            
            # Mark as visited and add to path
            if (r, c) not in visited_subcells:
                visited_subcells.add((r, c))
                path.append((r, c))
            
            # Determine which major cell this subcell belongs to
            major_r = r // self.subcell_per_cell
            major_c = c // self.subcell_per_cell
            
            # Get MST node for this major cell
            if (major_r, major_c) not in cell_to_node:
                continue
            
            node_idx = cell_to_node[(major_r, major_c)]
            
            # Explicit preferred neighbor exploration in anticlockwise order: LEFT, DOWN, RIGHT, UP
            directions = [(0, -1), (1, 0), (0, 1), (-1, 0)]
            to_push = []

            for dr, dc in directions:
                nr, nc = r + dr, c + dc

                # Check bounds
                if not (0 <= nr < self.total_grid_y and 0 <= nc < self.total_grid_x):
                    continue

                # Check if neighbor subcell is free
                if self.subcell_status(nr, nc) != 'free':
                    continue

                neighbor_major_r = nr // self.subcell_per_cell
                neighbor_major_c = nc // self.subcell_per_cell

                # If moving within same major cell allow movement
                if (major_r, major_c) == (neighbor_major_r, neighbor_major_c):
                    if (nr, nc) not in visited_subcells:
                        to_push.append((nr, nc))
                else:
                    # Moving to adjacent major cell: ensure it's present in component & MST edge exists
                    if (neighbor_major_r, neighbor_major_c) in cell_to_node:
                        neighbor_node = cell_to_node[(neighbor_major_r, neighbor_major_c)]
                        edge_exists = (node_idx in self.mst_adjacency and neighbor_node in self.mst_adjacency[node_idx])
                        if edge_exists and self.check_border_passable(r, c, nr, nc):
                            if (nr, nc) not in visited_subcells:
                                to_push.append((nr, nc))

            # Push onto stack in reverse so the first direction above is processed first (LIFO)
            for neighbor in reversed(to_push):
                stack.append((neighbor[0], neighbor[1], r, c))
        
        return path

    def check_border_passable(self, r1, c1, r2, c2):
        """Check if moving from subcell (r1,c1) to (r2,c2) is physically possible.
        This prevents moving through walls/obstacles between major cells."""
        
        # If moving horizontally (same row, different column)
        if r1 == r2:
            # Check all subcells between c1 and c2
            start_c = min(c1, c2)
            end_c = max(c1, c2)
            for c in range(start_c, end_c + 1):
                if self.subcell_status(r1, c) != 'free':
                    return False
        
        # If moving vertically (same column, different row)
        elif c1 == c2:
            # Check all subcells between r1 and r2
            start_r = min(r1, r2)
            end_r = max(r1, r2)
            for r in range(start_r, end_r + 1):
                if self.subcell_status(r, c1) != 'free':
                    return False
        
        return True

    def generate_major_edges(self):
        edges = []
        for r in range(self.main_grid_size_y):
            for c in range(self.main_grid_size_x):
                # Skip non-free cells
                if (r, c) not in self.major_node_indices:
                    continue
                    
                idx = self.major_node_indices[(r, c)]
                
                # Check right neighbor - only if physically connected
                if c + 1 < self.main_grid_size_x:
                    if (r, c + 1) in self.major_node_indices:
                        # Check if the border between these cells is passable
                        if self.check_major_cell_connection(r, c, r, c + 1):
                            nidx = self.major_node_indices[(r, c + 1)]
                            edges.append((1.0, idx, nidx))  # weight, from, to
                
                # Check down neighbor - only if physically connected
                if r + 1 < self.main_grid_size_y:
                    if (r + 1, c) in self.major_node_indices:
                        # Check if the border between these cells is passable
                        if self.check_major_cell_connection(r, c, r + 1, c):
                            nidx = self.major_node_indices[(r + 1, c)]
                            edges.append((1.0, idx, nidx))
        
        return edges

    def check_major_cell_connection(self, r1, c1, r2, c2):
        """Check if two adjacent major cells are physically connected (no wall between them)."""
        if r1 == r2:  # Same row, different columns (horizontal neighbors)
            # Check the border between these cells
            border_r_start = r1 * self.subcell_per_cell
            border_r_end = border_r_start + self.subcell_per_cell
            
            # Check subcells on the right border of first cell and left border of second cell
            for offset in range(self.subcell_per_cell):
                # Check right border of first cell
                if self.subcell_status(border_r_start + offset, c1 * self.subcell_per_cell + self.subcell_per_cell - 1) != 'free':
                    return False
                # Check left border of second cell
                if self.subcell_status(border_r_start + offset, c2 * self.subcell_per_cell) != 'free':
                    return False
        
        elif c1 == c2:  # Same column, different rows (vertical neighbors)
            # Check the border between these cells
            border_c_start = c1 * self.subcell_per_cell
            border_c_end = border_c_start + self.subcell_per_cell
            
            # Check subcells on the bottom border of first cell and top border of second cell
            for offset in range(self.subcell_per_cell):
                # Check bottom border of first cell
                if self.subcell_status(r1 * self.subcell_per_cell + self.subcell_per_cell - 1, border_c_start + offset) != 'free':
                    return False
                # Check top border of second cell
                if self.subcell_status(r2 * self.subcell_per_cell, border_c_start + offset) != 'free':
                    return False
        
        return True

    # -------------------------
    # Kruskal MST for major cells
    # -------------------------
    def kruskal_mst_major(self):
        if not self.major_edges:
            return []
            
        edges = sorted(self.major_edges, key=lambda x: x[0])
        ds = DisjointSet(len(self.major_nodes))
        mst = []
        for w, u, v in edges:
            if ds.union(u, v):
                mst.append((u, v))
        
        # Check if we have multiple components
        components = {}
        for i in range(len(self.major_nodes)):
            root = ds.find(i)
            if root not in components:
                components[root] = []
            components[root].append(i)
        
        if len(components) > 1:
            self.get_logger().warning(f"MST has {len(components)} disconnected components")
        
        return mst

    # -------------------------
    # Convert subcell to PoseStamped
    # -------------------------
    def subcell_to_pose(self, idx):
        row, col = self.path[idx]
        x = self.origin_x + (col + 0.5) * self.subcell_size
        y = self.origin_y + (row + 0.5) * self.subcell_size
        if idx < len(self.path) - 1:
            nr, nc = self.path[idx + 1]
            dx = (nc - col) * self.subcell_size
            dy = (nr - row) * self.subcell_size
            yaw = math.atan2(dy, dx)
        else:
            yaw = 0.0
        qz = math.sin(yaw / 2.0)
        qw = math.cos(yaw / 2.0)
        pose = PoseStamped()
        pose.header.frame_id = 'map'
        pose.header.stamp = self.get_clock().now().to_msg()
        pose.pose.position.x = x
        pose.pose.position.y = y
        pose.pose.position.z = 0.0
        pose.pose.orientation.z = qz
        pose.pose.orientation.w = qw
        return pose

    # -------------------------
    # Send next waypoint to Nav2
    # -------------------------
    def send_next_pose(self, idx):
        if idx >= len(self.path):
            self.get_logger().info("Finished coverage of all subcells!")
            return
        pose = self.subcell_to_pose(idx)
        row, col = self.path[idx]
        self.get_logger().info(f"Sending waypoint {idx+1}/{len(self.path)}: row={row}, col={col}")
        goal_msg = NavigateToPose.Goal()
        goal_msg.pose = pose
        self.nav_client.wait_for_server()
        send_goal_future = self.nav_client.send_goal_async(goal_msg)
        send_goal_future.add_done_callback(lambda f, i=idx: self.goal_response(f, i))

    def goal_response(self, future, idx):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().error("Goal rejected!")
            return
        get_result_future = goal_handle.get_result_async()
        get_result_future.add_done_callback(lambda f, i=idx: self.result_callback(f, i))

    def result_callback(self, future, idx):
        self.send_next_pose(idx + 1)

    # -------------------------
    # Visualization markers in RViz
    # -------------------------
    def publish_markers(self):
        markers = MarkerArray()
        marker_id = 0

        # ---------- Subcells (thin green outline) ----------
        for r in range(self.total_grid_y):
            for c in range(self.total_grid_x):
                m = Marker()
                m.header.frame_id = "map"
                m.type = Marker.LINE_STRIP
                m.ns = "subcells"
                m.action = Marker.ADD
                m.id = marker_id
                marker_id += 1
                m.scale.x = 0.01  # thin
                status = self.subcell_status(r, c)
                if status == 'obstacle':
                    m.color = ColorRGBA(r=1.0, g=0.0, b=0.0, a=0.6)
                elif status == 'invalid':
                    m.color = ColorRGBA(r=0.6, g=0.6, b=0.6, a=0.35)
                else:
                    m.color = ColorRGBA(r=0.0, g=0.85, b=0.0, a=0.25)  # thin green fill-ish outline
                x = self.origin_x + c * self.subcell_size
                y = self.origin_y + r * self.subcell_size
                pts = [
                    Point(x=x, y=y, z=0.0),
                    Point(x=x+self.subcell_size, y=y, z=0.0),
                    Point(x=x+self.subcell_size, y=y+self.subcell_size, z=0.0),
                    Point(x=x, y=y+self.subcell_size, z=0.0),
                    Point(x=x, y=y, z=0.0)
                ]
                m.points = pts
                markers.markers.append(m)

        # ---------- Major cells (thick blue outline for free, red filled cube for occupied, grey for invalid) ----------
        # also prepare a list of major cell centers for MST labeling
        major_centers = []  # list of (x,y)
        for mr in range(self.main_grid_size_y):
            for mc in range(self.main_grid_size_x):
                status = self.major_cell_occupancy[mr][mc]
                # Outline marker
                out = Marker()
                out.header.frame_id = "map"
                out.type = Marker.LINE_STRIP
                out.ns = "major_cells"
                out.action = Marker.ADD
                out.id = marker_id
                marker_id += 1
                out.scale.x = 0.05  # thick outline

                x = self.origin_x + mc * self.cell_size
                y = self.origin_y + mr * self.cell_size
                pts = [
                    Point(x=x, y=y, z=0.0),
                    Point(x=x+self.cell_size, y=y, z=0.0),
                    Point(x=x+self.cell_size, y=y+self.cell_size, z=0.0),
                    Point(x=x, y=y+self.cell_size, z=0.0),
                    Point(x=x, y=y, z=0.0)
                ]
                out.points = pts

                if status == 'obstacle':
                    # show red outline and also a semi-transparent cube filling for clarity
                    out.color = ColorRGBA(r=1.0, g=0.0, b=0.0, a=0.9)
                    # cube fill
                    cube = Marker()
                    cube.header.frame_id = "map"
                    cube.type = Marker.CUBE
                    cube.ns = "major_filled"
                    cube.action = Marker.ADD
                    cube.id = marker_id
                    marker_id += 1
                    cx = x + 0.5 * self.cell_size
                    cy = y + 0.5 * self.cell_size
                    cube.pose.position.x = cx
                    cube.pose.position.y = cy
                    cube.pose.position.z = 0.0
                    cube.scale.x = self.cell_size
                    cube.scale.y = self.cell_size
                    cube.scale.z = 0.01
                    cube.color = ColorRGBA(r=1.0, g=0.0, b=0.0, a=0.35)
                    markers.markers.append(cube)
                    major_centers.append((cx, cy))
                elif status == 'invalid':
                    out.color = ColorRGBA(r=0.6, g=0.6, b=0.6, a=0.9)
                    cx = x + 0.5 * self.cell_size
                    cy = y + 0.5 * self.cell_size
                    major_centers.append((cx, cy))
                else:
                    out.color = ColorRGBA(r=0.0, g=0.35, b=1.0, a=0.95)  # thick blue outline for free major cells
                    cx = x + 0.5 * self.cell_size
                    cy = y + 0.5 * self.cell_size
                    major_centers.append((cx, cy))

                markers.markers.append(out)

                # Add text label for major cell coords (small)
                text = Marker()
                text.header.frame_id = "map"
                text.type = Marker.TEXT_VIEW_FACING
                text.ns = "major_labels"
                text.action = Marker.ADD
                text.id = marker_id
                marker_id += 1
                text.scale.z = min(0.12, self.cell_size * 0.25)  # text size relative to cell
                text.color = ColorRGBA(r=0.9, g=0.9, b=0.9, a=0.9)
                text.pose.position.x = cx
                text.pose.position.y = cy
                text.pose.position.z = 0.02
                # show grid coordinates
                text.text = f"{mr},{mc}"
                markers.markers.append(text)

        # ---------- MST edges with arrows to show direction ----------
        if hasattr(self, 'major_mst_edges') and self.major_mst_edges:
            # First show lines
            mst_marker = Marker()
            mst_marker.header.frame_id = "map"
            mst_marker.type = Marker.LINE_LIST
            mst_marker.ns = "mst_edges"
            mst_marker.action = Marker.ADD
            mst_marker.id = marker_id
            marker_id += 1
            mst_marker.scale.x = 0.08  # thicker
            mst_marker.color = ColorRGBA(r=1.0, g=0.0, b=0.0, a=1.0)  # red
            
            # Show arrows at midpoints
            arrow_markers = []
            
            for (u, v) in self.major_mst_edges:
                if u < 0 or v < 0 or u >= len(self.major_nodes) or v >= len(self.major_nodes):
                    continue
                
                y1, x1 = self.major_nodes[u]
                y2, x2 = self.major_nodes[v]
                
                # Add line
                p1 = Point(x=x1, y=y1, z=0.02)
                p2 = Point(x=x2, y=y2, z=0.02)
                mst_marker.points.append(p1)
                mst_marker.points.append(p2)
                
                # Add arrow at midpoint
                arrow = Marker()
                arrow.header.frame_id = "map"
                arrow.type = Marker.ARROW
                arrow.ns = "mst_arrows"
                arrow.action = Marker.ADD
                arrow.id = marker_id
                marker_id += 1
                
                # Calculate midpoint and orientation
                mid_x = (x1 + x2) / 2
                mid_y = (y1 + y2) / 2
                dx = x2 - x1
                dy = y2 - y1
                
                arrow.pose.position.x = mid_x
                arrow.pose.position.y = mid_y
                arrow.pose.position.z = 0.02
                
                # Calculate yaw from dx, dy
                yaw = math.atan2(dy, dx)
                qz = math.sin(yaw / 2.0)
                qw = math.cos(yaw / 2.0)
                
                arrow.pose.orientation.z = qz
                arrow.pose.orientation.w = qw
                
                arrow.scale.x = self.cell_size * 0.6  # arrow length
                arrow.scale.y = 0.1  # arrow width
                arrow.scale.z = 0.1  # arrow height
                arrow.color = ColorRGBA(r=1.0, g=0.0, b=0.0, a=1.0)
                
                arrow_markers.append(arrow)
            
            markers.markers.append(mst_marker)
            for arrow in arrow_markers:
                markers.markers.append(arrow)

        # ---------- STC traversal path with direction indicators ----------
        # DRAW PER-COMPONENT PATHS to avoid RViz drawing a straight connecting line between disconnected components
        if hasattr(self, 'component_paths') and self.component_paths:
            # draw each connected component as its own LINE_STRIP to avoid long connector lines
            for comp_idx, comp_path in enumerate(self.component_paths):
                comp_marker = Marker()
                comp_marker.header.frame_id = "map"
                comp_marker.type = Marker.LINE_STRIP
                comp_marker.ns = f"stc_path_{comp_idx}"
                comp_marker.action = Marker.ADD
                comp_marker.id = marker_id
                marker_id += 1
                comp_marker.scale.x = 0.08  # slightly thinner per-component line
                comp_marker.color = ColorRGBA(r=0.0, g=1.0, b=0.0, a=1.0)  # green

                # Add direction arrows at key points for this component
                if len(comp_path) > 0:
                    arrow_interval = max(1, len(comp_path) // 8)  # show ~8 arrows per component

                    for i, (r, c) in enumerate(comp_path):
                        px = self.origin_x + (c + 0.5) * self.subcell_size
                        py = self.origin_y + (r + 0.5) * self.subcell_size
                        comp_marker.points.append(Point(x=px, y=py, z=0.03))

                        if i % arrow_interval == 0 and i < len(comp_path) - 1:
                            next_r, next_c = comp_path[i + 1]
                            next_px = self.origin_x + (next_c + 0.5) * self.subcell_size
                            next_py = self.origin_y + (next_r + 0.5) * self.subcell_size

                            dx = next_px - px
                            dy = next_py - py
                            yaw = math.atan2(dy, dx)

                            arrow = Marker()
                            arrow.header.frame_id = "map"
                            arrow.type = Marker.ARROW
                            arrow.ns = f"path_arrows_{comp_idx}"
                            arrow.action = Marker.ADD
                            arrow.id = marker_id
                            marker_id += 1

                            arrow.pose.position.x = px
                            arrow.pose.position.y = py
                            arrow.pose.position.z = 0.03

                            qz = math.sin(yaw / 2.0)
                            qw = math.cos(yaw / 2.0)
                            arrow.pose.orientation.z = qz
                            arrow.pose.orientation.w = qw

                            arrow.scale.x = self.subcell_size * 0.8
                            arrow.scale.y = 0.05
                            arrow.scale.z = 0.05
                            arrow.color = ColorRGBA(r=1.0, g=1.0, b=0.0, a=1.0)  # yellow arrows

                            markers.markers.append(arrow)

                markers.markers.append(comp_marker)

        # ---------- Robot pose marker (small green cube) ----------
        if self.robot_pose_received:
            robot_m = Marker()
            robot_m.header.frame_id = "map"
            robot_m.type = Marker.CUBE
            robot_m.ns = "robot"
            robot_m.action = Marker.ADD
            robot_m.id = marker_id
            marker_id += 1
            robot_m.scale.x = 0.18
            robot_m.scale.y = 0.18
            robot_m.scale.z = 0.02
            robot_m.pose.position.x = self.robot_pose_x
            robot_m.pose.position.y = self.robot_pose_y
            robot_m.pose.position.z = 0.01
            robot_m.color = ColorRGBA(r=0.0, g=1.0, b=0.0, a=1.0)
            markers.markers.append(robot_m)

        self.get_logger().info(
            f"Publishing MarkerArray with {len(markers.markers)} markers"
        )
        self.marker_pub.publish(markers)
        self.get_logger().info("Published Kruskal MST STC markers!")

# -------------------------
# Main
# -------------------------
def main(args=None):
    rclpy.init(args=args)
    node = KruskalSTCNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()
