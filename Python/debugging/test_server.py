import zmq
import json
import time


def generate_mock_pose(frame_number: int, num_joints: int = 10) -> dict:
    return {
        "JointLocalPositions": [{"x": 0.0, "y": 1.0, "z": 0.0} for _ in range(num_joints)],

        "JointLocalRotations": [{"value": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0}} for _ in range(num_joints)],

        "JointLocalVelocities": [{"x": 0.0, "y": 0.0, "z": 0.0} for _ in range(num_joints)],
        "JointLocalAngularVelocities": [{"x": 0.0, "y": 0.0, "z": 0.0} for _ in range(num_joints)],
        "LeftFootContact": True,
        "RightFootContact": False
    }

def main():
    context = zmq.Context()
    socket = context.socket(zmq.REP)
    socket.bind("tcp://*:5555")

    print("Python ZeroMQ server listening on port 5555...")

    while True:
        # Read raw bytes first so we don't crash on bad queued data
        raw_message = socket.recv()

        try:
            message = raw_message.decode('utf-8')
            print(f"Received request for frame: {message}")
        except UnicodeDecodeError:
            print(f"Ignored invalid/leftover binary data: {raw_message}")
            socket.send_string(json.dumps({"error": "Invalid string encoding"}))
            continue

        try:
            frame_number = int(message)
            pose_data = generate_mock_pose(frame_number, 23)
            # Send serialized JSON back to Unity
            socket.send_string(json.dumps(pose_data))
        except ValueError:
            print(f"Invalid frame number format: {message}")
            socket.send_string(json.dumps({"error": "Invalid frame number"}))




if __name__ == "__main__":
    main()