from controller import Supervisor, Motor, PositionSensor
import math
import time

# Global parameters
TIME_STEP = 32  # Simulation time step in ms
# Initial motor positions in degrees, from my_controllerA4.py
# These are the "center" positions for each joint.
initial_motor_positions = [-45, 90, -110, 45, 90, -110, -45, -90, 110, 45, -90, 110, 0, -90, 110, 0, 90, -110]
# The amount to move each joint in degrees from its initial position
MOVE_ANGLE = 20

def main():
    # Initialize the Supervisor robot
    robot = Supervisor()

    # Get the robot node
    robot_node = robot.getFromDef("IRSL-XR06-01")
    if robot_node is None:
        print("Error: Robot node 'IRSL-XR06-01' not found.")
        return

    # Get all motors and position sensors
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

    print("Setting initial motor positions...")
    # Set all motors to their initial positions
    for i in range(18):
        motors[i].setPosition(math.radians(initial_motor_positions[i]))
    # Let the robot stabilize
    for _ in range(10):
        robot.step(TIME_STEP)

    print("Starting joint movement test...")
    # Iterate through each joint and move it
    for i in range(18):
        print(f"\nMoving motor{i+1}...")

        # Move +20 degrees from initial position
        target_pos_plus = initial_motor_positions[i] + MOVE_ANGLE
        print(f"  Moving motor{i+1} to {target_pos_plus} degrees.")
        motors[i].setPosition(math.radians(target_pos_plus))
        for _ in range(20): # Wait for movement
            robot.step(TIME_STEP)
        current_angle = math.degrees(position_sensors[i].getValue())
        print(f"  motor{i+1} current position: {current_angle:.2f} degrees.")

        # Move -20 degrees from initial position
        target_pos_minus = initial_motor_positions[i] - MOVE_ANGLE
        print(f"  Moving motor{i+1} to {target_pos_minus} degrees.")
        motors[i].setPosition(math.radians(target_pos_minus))
        for _ in range(20): # Wait for movement
            robot.step(TIME_STEP)
        current_angle = math.degrees(position_sensors[i].getValue())
        print(f"  motor{i+1} current position: {current_angle:.2f} degrees.")

        # Return to initial position
        print(f"  Returning motor{i+1} to initial position ({initial_motor_positions[i]} degrees).")
        motors[i].setPosition(math.radians(initial_motor_positions[i]))
        for _ in range(20): # Wait for movement
            robot.step(TIME_STEP)
        current_angle = math.degrees(position_sensors[i].getValue())
        print(f"  motor{i+1} current position: {current_angle:.2f} degrees.")

    print("\nJoint movement test completed.")

if __name__ == "__main__":
    main()
