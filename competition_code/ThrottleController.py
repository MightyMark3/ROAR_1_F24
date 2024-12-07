import numpy as np
import math
from collections import deque
import roar_py_interface


def distance_p_to_p(
    p1: roar_py_interface.RoarPyWaypoint, p2: roar_py_interface.RoarPyWaypoint
):
    return np.linalg.norm(p2.location[:2] - p1.location[:2])

class SpeedData:
    def __init__(
        self, distance_to_section, current_speed, current_section, current_ind, next_waypoint_ind, target_speed, recommended_speed
    ):
        self.radius = 0
        self.radius_name = "r"
        self.current_speed = current_speed
        self.current_section = current_section
        self.distance_to_section = distance_to_section
        self.target_speed_at_distance = target_speed
        self.target_speed_now_mu1 = 0
        self.recommended_speed_now = recommended_speed
        self.speed_diff = current_speed - recommended_speed
        self.current_index = current_ind
        self.next_waypoint_ind = next_waypoint_ind
        self.distance_to_min_r = 0


class DebugInfo:
    def __init__(self, tick_counter, current_speed, current_location, current_section, current_ind):
        self.tick = tick_counter
        self.x = current_location[0]
        self.y = current_location[1]
        self.z = current_location[2]
        self.current_speed = current_speed
        self.current_section = current_section
        self.current_ind = current_ind
        self.throttle = 0
        self.brake = 0
        self.radius = 0
        self.radius_name = "r"
        self.distance_to_section = 0
        self.target_speed_at_distance = 0
        self.target_speed_now = 0
        self.target_speed_now_mu1 = 0
        # situation which produced this throttle/brake
        self.s_name = "s"
        self.distance_to_min_r = 1000

    def copy_from_speed_data(self, speed_data, s_name):
        self.radius = speed_data.radius
        self.radius_name = speed_data.radius_name
        self.distance_to_section = speed_data.distance_to_section
        self.target_speed_at_distance = speed_data.target_speed_at_distance
        self.target_speed_now = speed_data.recommended_speed_now
        self.target_speed_now_mu1 = speed_data.target_speed_now_mu1
        self.s_name = s_name
        self.distance_to_min_r = speed_data.distance_to_min_r


class ThrottleController:
    display_debug = False
    debug_strings = deque(maxlen=1000)

    def __init__(self, waypoints):
        self.waypoints = waypoints
        self.max_radius = 10000
        self.max_speed = 300
        self.intended_target_distance = [0, 30, 60, 90, 120, 140, 170]
        self.target_distance = [0, 30, 60, 90, 120, 150, 180]
        self.close_index = 0
        self.mid_index = 1
        self.far_index = 2
        self.tick_counter = 0
        self.previous_speed = 1.0
        self.brake_ticks = 0
        self.brake_count = 0

        # for testing how fast the car stops
        self.brake_test_counter = 0
        self.brake_test_in_progress = False

        self.test_speed = 184
        self.prev_section = 0
        self.radius_at_waypoint = self.precompute_radius()
        self.indexes_with_low_r = self.find_low_radius()


    def __del__(self):
        print("done")

    def run(self, current_location, current_speed, current_section, current_ind, next_waypoint_ind):
        self.tick_counter += 1
        waypoints = (self.waypoints * 2)[next_waypoint_ind : next_waypoint_ind + 500]
        debug_info = DebugInfo(self.tick_counter, current_speed, current_location, current_section, current_ind)
        throttle, brake = self.get_throttle_and_brake(
            current_location, current_speed, current_section, waypoints, debug_info, current_ind, next_waypoint_ind
        )
        gear = max(1, int(current_speed / 60))
        if throttle < 0:
            gear = -1

        self.previous_speed = current_speed

        # if brake > 0:
        #     self.brake_count += 1
        #     if self.brake_count == 1: 
        #         print(f"brake in section {current_section} {next_waypoint_ind} num_br_t {self.brake_ticks} sp {current_speed:.1f}")
        # else:
        #     if self.brake_count > 0:
        #         print(f"brake in section {current_section} {next_waypoint_ind} finished {self.brake_count} sp {current_speed:.1f}")
        #     self.brake_count = 0

        if self.brake_ticks > 0 and brake > 0:
            self.brake_ticks -= 1

        debug_info.throttle = throttle
        debug_info.brake = brake
        self.print_debug_info(debug_info, next_waypoint_ind)
        return throttle, brake, gear
    
    def print_debug_info(self, d, next_waypoint_ind):
      if d.current_section in [9]:
        print(f"p {d.tick:>4}, {d.current_speed: >6.2f}, " +
              f"{d.target_speed_now: >6.2f}, " +
              f"{d.throttle:.2f}, {d.brake:.2f}, " +
              f"{d.distance_to_min_r:5.1f}, " +
              f"{d.s_name:>3}, {d.x:7.2f}, {d.y:7.2f}, " +
            #   f"{d.z:7.2f}, " +
            #   f"{d.target_speed_now_mu1: >6.2f}, " +
              f"{d.radius: >5.1f}, " +
              f"{d.radius_name:>2}, " +
            #   f"{d.distance_to_section:3.0f}, {d.target_speed_at_distance:.2f}, " +
              f"{d.current_section}, {d.current_ind:>4}, {next_waypoint_ind:>4}")
        self.distance_to_section = 0
        self.target_speed_at_distance = 0

    def get_throttle_and_brake(
        self, current_location, current_speed, current_section, waypoints, debug_info, current_ind, next_waypoint_ind
    ):
        nextWaypoint = self.get_next_interesting_waypoints(current_location, waypoints)
        close_distance = self.target_distance[self.close_index] + 3
        mid_distance = self.target_distance[self.mid_index]
        far_distance = self.target_distance[self.far_index]

        sd1 = self.compute_speed_data("r1",
            nextWaypoint[self.close_index : self.close_index + 3], close_distance, current_section, current_speed, current_ind, next_waypoint_ind)
        sd2 = self.compute_speed_data("r2",
            nextWaypoint[self.mid_index : self.mid_index + 3], mid_distance, current_section, current_speed, current_ind, next_waypoint_ind)
        sd3 = self.compute_speed_data("r3",
            nextWaypoint[self.far_index : self.far_index + 3], far_distance, current_section, current_speed, current_ind, next_waypoint_ind)

        speed_data = []
        if not (sd1.radius < 80 and current_section == 6):
            speed_data.append(sd1)
        speed_data.append(sd2)
        speed_data.append(sd3)

        if current_speed > 100:
            # at high speed use larger spacing between points to look further ahead and detect wide turns.
            if current_section != 9:
                r4_waypoints = [
                    nextWaypoint[self.mid_index],
                    nextWaypoint[self.mid_index + 2],
                    nextWaypoint[self.mid_index + 4],
                ]
                speed_data.append(
                    self.compute_speed_data("r4", r4_waypoints, close_distance, current_section, current_speed, current_ind, next_waypoint_ind)
                )

            r5_waypoints = [
                nextWaypoint[self.close_index],
                nextWaypoint[self.close_index + 3],
                nextWaypoint[self.close_index + 6],
            ]
            speed_data.append(
                self.compute_speed_data("r5", r5_waypoints, close_distance, current_section, current_speed, current_ind, next_waypoint_ind)
            )

        update = self.select_speed(speed_data)
        update.distance_to_min_r = self.get_distance_to_point(current_location, current_ind)

        throttle, brake, s_name = self.speed_data_to_throttle_and_brake(update)
        throttle = max(0, min(throttle, 1.0))
        debug_info.copy_from_speed_data(update, s_name)
        return throttle, brake

    def compute_speed_data(self, radius_name, waypoints, distance, current_section, current_speed, current_ind, next_waypoint_ind):
        turn_radius = self.get_radius(waypoints, current_section)
        return self.speed_for_turn_new(distance, turn_radius, radius_name, current_section, current_speed, current_ind, next_waypoint_ind)

    def speed_data_to_throttle_and_brake(self, speed_data: SpeedData):
        """
        Converts speed data into throttle and brake values
        """
        percent_of_max = speed_data.current_speed / speed_data.recommended_speed_now
        speed_excess = speed_data.current_speed - speed_data.recommended_speed_now
        # speed_change_per_tick = 3.2  # Speed decrease in kph per tick
        speed_change_per_tick = 2.4  # Speed decrease in kph per tick
        percent_change_per_tick = 0.075  # speed drop for one time-tick of braking
        true_percent_change_per_tick = round(
            speed_change_per_tick / (speed_data.current_speed + 0.001), 5
        )
        speed_up_threshold = 0.9
        throttle_decrease_multiple = 0.7
        throttle_increase_multiple = 1.25
        brake_threshold_multiplier = 1.0
        percent_speed_change = (speed_data.current_speed - self.previous_speed) / (
            self.previous_speed + 0.0001
        )  # avoid division by zero
        speed_change = round(speed_data.current_speed - self.previous_speed, 3)
        speed_decrease = self.previous_speed - speed_data.current_speed

        br_value = 1.0
        br_value_low = 1.0
        br_release_speed = 0
        speed_excess_threshold = 0
        if speed_data.current_section == 4:
            br_value = 0.9
            speed_excess_threshold = 15
            br_value_low = br_value / 2
            br_release_speed = 167  # 170-, 165+,  
        if speed_data.current_section == 6:
            br_value = 0.5
            speed_excess_threshold = 20
            br_value_low = br_value / 3
            br_release_speed = 213 # 220-, 210+, 215+, 215-, 212+, 213+ 
        if speed_data.current_section == 9:
            br_value = 1.0
            speed_excess_threshold = 30
            br_value_low = br_value / 2
            br_release_speed = 130 # 150-, 140-, 130+, 135-, 130+

        if percent_of_max > 1:
            # Consider slowing down
            if speed_data.current_speed > 200:  # Brake earlier at higher speeds
                brake_threshold_multiplier = 0.9
            

            if speed_data.current_speed < br_release_speed:
                return 1, 0, "sbr"  # break release

            if speed_data.current_section in [4, 6, 9] and speed_excess < speed_excess_threshold and speed_decrease > 2.0:
                # reduce break
                return 0, br_value_low, "slb"

            if percent_of_max > 1 + (brake_threshold_multiplier * true_percent_change_per_tick):
                if self.brake_ticks > 0:
                    return 0, br_value, "s1"

                # if speed is not decreasing fast, hit the brake.
                if self.brake_ticks <= 0 and speed_change < 1.5:
                    # start braking, and set for how many ticks to brake
                    self.brake_ticks = \
                        round((speed_data.current_speed - speed_data.recommended_speed_now) / speed_change_per_tick) + 2
                    return 0, 1, "s2"

                else:
                    # speed is already dropping fast, ok to throttle because the effect of throttle is delayed
                    self.brake_ticks = 0  # done slowing down. clear brake_ticks
                    return 1, 0, "s3"
            else:
                if speed_change >= 1.5:
                    # speed is already dropping fast, ok to throttle because the effect of throttle is delayed
                    self.brake_ticks = 0  # done slowing down. clear brake_ticks
                    return 1, 0, "s4"

                # TODO: Try to get rid of coasting. Unnecessary idle time that could be spent speeding up or slowing down
                throttle_to_maintain = self.get_throttle_to_maintain_speed(speed_data.current_speed)

                if percent_of_max > 1.02 or percent_speed_change > (
                    -true_percent_change_per_tick / 2
                ):
                    return (
                        throttle_to_maintain * throttle_decrease_multiple,
                        0, "s5"
                    )  # coast, to slow down
                else:
                    return throttle_to_maintain, 0, "s6"
        else:
            self.brake_ticks = 0  # done slowing down. clear brake_ticks
            # Speed up
            if speed_change >= 1.5:
                # speed is dropping fast, ok to throttle because the effect of throttle is delayed
                return 1, 0, "s7"
            if percent_of_max < speed_up_threshold:
                return 1, 0, "s8"
            throttle_to_maintain = self.get_throttle_to_maintain_speed(speed_data.current_speed)
            if percent_of_max < 0.98 or true_percent_change_per_tick < -0.01:
                return throttle_to_maintain * throttle_increase_multiple, 0, "s9"
            else:
                return throttle_to_maintain, 0, "s10"

    # used to detect when speed is dropping due to brakes applied earlier. speed delta has a steep negative slope.
    def isSpeedDroppingFast(self, percent_change_per_tick: float, current_speed):
        """
        Detects if the speed of the car is dropping quickly.
        Returns true if the speed is dropping fast
        """
        percent_speed_change = (current_speed - self.previous_speed) / (
            self.previous_speed + 0.0001
        )  # avoid division by zero
        return percent_speed_change < (-percent_change_per_tick / 2)

    # find speed_data with smallest recommended speed
    def select_speed(self, speed_data: [SpeedData]):
        """
        Selects the smallest speed out of the speeds provided
        """
        min_speed = 1000
        index_of_min_speed = -1
        for i, sd in enumerate(speed_data):
            if sd.recommended_speed_now < min_speed:
                min_speed = sd.recommended_speed_now
                index_of_min_speed = i

        if index_of_min_speed != -1:
            return speed_data[index_of_min_speed]
        else:
            return speed_data[0]

    def get_throttle_to_maintain_speed(self, current_speed: float):
        """
        Returns a throttle value to maintain the current speed
        """
        throttle = 0.75 + current_speed / 500
        return throttle

    def speed_for_turn_new(
        self, distance: float, radius: float, radius_name: str, current_section: int, current_speed: float, current_ind, next_waypoint_ind
    ):
        """Generates a SpeedData object with the target speed for the far

        Args:
            distance (float): Distance from the start of the curve
            target_speed (float): Target speed of the curve
            current_speed (float): Current speed of the car

        Returns:
            SpeedData: A SpeedData object containing the distance to the corner, current speed, target speed, and max speed
        """
        # Takes in a target speed and distance and produces a speed that the car should target. Returns a SpeedData object
        target_speed = self.get_target_speed(radius, current_section)
        max_speed = self.get_max_speed(target_speed, distance)

        target_speed_mu1 = self.get_target_speed(radius, current_section, 1.0)
        max_speed_mu1 = self.get_max_speed(target_speed_mu1, distance)

        speed_data = SpeedData(distance, current_speed, current_section, current_ind, next_waypoint_ind, target_speed, max_speed)
        speed_data.radius = radius
        speed_data.radius_name = radius_name
        speed_data.target_speed_now_mu1 = max_speed_mu1
        return speed_data
    
    def get_max_speed(self, target_speed, distance):
        d = (1 / 675) * (target_speed**2) + distance
        return math.sqrt(825 * d)

    def get_next_interesting_waypoints(self, current_location, more_waypoints):
        """Returns a list of waypoints that are approximately as far as specified in intended_target_distance from the current location

        Args:
            current_location (roar_py_interface.RoarPyWaypoint): The current location of the car
            more_waypoints ([roar_py_interface.RoarPyWaypoint]): A list of waypoints

        Returns:
            [roar_py_interface.RoarPyWaypoint]: A list of waypoints within specified distances of the car
        """
        # Returns a list of waypoints that are approximately as far as the given in intended_target_distance from the current location

        # return a list of points with distances approximately as given
        # in intended_target_distance[] from the current location.
        points = []
        dist = []  # for debugging
        start = roar_py_interface.RoarPyWaypoint(
            current_location, np.ndarray([0, 0, 0]), 0.0
        )
        # start = self.agent.vehicle.transform
        points.append(start)
        curr_dist = 0
        num_points = 0
        for p in more_waypoints:
            end = p
            num_points += 1
            curr_dist += distance_p_to_p(start, end)
            if curr_dist > self.intended_target_distance[len(points)]:
                self.target_distance[len(points)] = curr_dist
                points.append(end)
                dist.append(curr_dist)
            start = end
            if len(points) >= len(self.target_distance):
                break

        return points

    def get_radius(self, wp: [roar_py_interface.RoarPyWaypoint], current_section=None):
        """Returns the radius of a curve given 3 waypoints using the Menger Curvature Formula

        Args:
            wp ([roar_py_interface.RoarPyWaypoint]): A list of 3 RoarPyWaypoints

        Returns:
            float: The radius of the curve made by the 3 given waypoints
        """

        point1 = (wp[0].location[0], wp[0].location[1])
        point2 = (wp[1].location[0], wp[1].location[1])
        point3 = (wp[2].location[0], wp[2].location[1])

        # Calculating length of all three sides
        len_side_1 = round(math.dist(point1, point2), 3)
        len_side_2 = round(math.dist(point2, point3), 3)
        len_side_3 = round(math.dist(point1, point3), 3)

        small_num = 2

        if len_side_1 < small_num or len_side_2 < small_num or len_side_3 < small_num:
            return self.max_radius

        # sp is semi-perimeter
        sp = (len_side_1 + len_side_2 + len_side_3) / 2

        # Calculating area using Herons formula
        area_squared = sp * (sp - len_side_1) * (sp - len_side_2) * (sp - len_side_3)
        if area_squared < small_num:
            return self.max_radius

        # Calculating curvature using Menger curvature formula
        radius = (len_side_1 * len_side_2 * len_side_3) / (4 * math.sqrt(area_squared))

        return radius

    def mu_for_section(self, current_section: int):
        mu = 2.4
        if current_section == 2:
            mu = 3.16
        if current_section == 3:
            mu = 3.15
            # mu = 3.0  # worked
        if current_section == 4:
            # mu = 2.4 # old
            mu = 2.4
        if current_section == 6:
            # mu = 3.1  # old
            mu = 3.3
        if current_section == 9:
            mu = 2.2  # old
        return mu

    def get_target_speed(self, radius: float, current_section: int, mu=None):
        """Returns a target speed based on the radius of the turn and the section it is in

        Args:
            radius (float): The calculated radius of the turn
            current_section (int): The current section of the track the car is in

        Returns:
            float: The maximum speed the car can go around the corner at
        """
        if radius >= self.max_radius:
            return self.max_speed

        if mu is None:
            mu = self.mu_for_section(current_section)
        target_speed = math.sqrt(mu * 9.81 * radius) * 3.6

        return max(20, min(target_speed, self.max_speed))  # clamp between 20 and max_speed

    def get_ind_of_next_min_radius(self, current_ind):
        """returns ind that could exceed len, need to apply modulo."""
        for i in range(current_ind, len(self.waypoints) * 2):
            ind = i % len(self.waypoints)
            if ind in self.indexes_with_low_r:
                return i
        return -1
    
    def get_distance_to_point(self, current_location, current_ind):
        ind_of_next_min_r = self.get_ind_of_next_min_radius(current_ind)
        if ind_of_next_min_r == -1:
            return 300

        min_location = self.waypoints[ ind_of_next_min_r % len(self.waypoints) ].location[:2]
        return np.linalg.norm(min_location - current_location[:2])

    def precompute_radius(self):
        # for each waypoint compute turn radius using 3 points
        #   (point which is 10 points before this point), 
        #   (this point), 
        #   (point which is 10 points after this points)
        dist = 10
        r_list = [self.max_radius] * len(self.waypoints)
        for i in range(0, len(self.waypoints)):
            s = (i - dist) % (len(self.waypoints))
            e = (i + dist) % (len(self.waypoints))
            radius = self.get_radius([self.waypoints[s], self.waypoints[i], self.waypoints[e]])
            r_list[i] = radius
        return r_list
    
    def min_index(self, start, end):
        min_r = 10000
        min_ind = 0
        for i in range(start, end+1):
            if self.radius_at_waypoint[i] < min_r:
                min_r = self.radius_at_waypoint[i]
                min_ind = i
        return min_ind
    
    def find_low_radius(self):
        dist = 10
        ind_list = []
        for i in range(0, len(self.waypoints)):
            s = (i - dist) % (len(self.waypoints))
            e = (i + dist) % (len(self.waypoints))
            # print(i, s, e, self.min_index(s, e))
            if self.radius_at_waypoint[i] < 90 \
                and self.min_index(s, e) == i:
                ind_list.append(i)
        return ind_list
