from controller import Supervisor, Motor, PositionSensor
import math
import time

# Global parameters
TIME_STEP = 32  # Simulation time step in ms
# The amount to move each joint in degrees from its initial position
MOVE_ANGLE = 20 # This variable is no longer used for movement, but kept for consistency if needed later.

# List of multiple sets of initial motor positions
all_initial_motor_positions_sets = [
    # Set 1: Modified positions (foot-closest joints adjusted)
    [-45, 90, -110, 45, 90, -110, -45, -90, 110, 45, -90, 110, 0, -90, 110, 0, 90, -110],
    # Set 2: Original positions
    [-45, 90, -130, 45, 90, -130, -45, -90, 130, 45, -90, 130, 0, -90, 130, 0, 90, -130]
]

def main():
    # Initialize the Supervisor robot
    robot = Supervisor()

    # Get the robot node
    robot_node = robot.getFromDef("IRSL-XR06-01")
    if robot_node is None:
        print("Error: Robot node 'IRSL-XR06-01' not found.")
        return

    # Get all motors and position sensors (using a loop as in original check_pos.py for maintainability)
    motors = []
    position_sensors = []
    for i in range(18):
        motor_name = f"motor{i+1}"
        ps_name = f"PS{i+1}"
        motor = robot.getMotor(motor_name)
        ps = robot.getPositionSensor(ps_name)
        ps.enable(TIME_STEP)
        motors.append(motor)
        position_sensors.append(ps)

    # Loop through each set of initial motor positions
    for set_idx, current_initial_motor_positions in enumerate(all_initial_motor_positions_sets):
        print(f"\n--- Executing test for Initial Motor Position Set {set_idx + 1} ---")
        print(f"Using initial positions: {current_initial_motor_positions}")

        print("Setting initial motor positions for this set...")
        # Set all motors to their initial positions for the current set
        for i in range(18):
            motors[i].setPosition(math.radians(current_initial_motor_positions[i]))
        # Let the robot stabilize
        for _ in range(50): # Increased stabilization time
            robot.step(TIME_STEP)

        print(f"Robot is now in the initial position for Set {set_idx + 1}.")
        # Optionally, you could add a longer pause here or wait for user input
        # For now, it will just proceed to the next set after stabilization.

    print("\nAll initial position tests completed.")

if __name__ == "__main__":
    main()