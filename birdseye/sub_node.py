from birdseye.camera.projection_model import ProjectionEngine, MeshBackend
from birdseye.camera.camera import SensorConfigLoader, PinholeCameraModel


class PoseResolver:
    def __init__(self, extrinsics):
        self.extrinsics = extrinsics

    def get_T_world_cam_tf2(self, cam_name, timestamp):
        T_world_ins = TF.lookup("world", "ins", t)
        T_ins_cam = self.extrinsics[cam_name]
        return T_world_ins @ T_ins_cam

    def get_T_cam_world_strobe(self):
        # TODO: complete
        return T_world_ins @ T_ins_cam


class ProjectionNode(Node):
    def __init__(self):
        self.engine = ProjectionEngine(...)
        self.pose = PoseResolver(...)
        self.keyframe = KeyframeSelector()

        self.pipelines = {
            cam: CameraPipeline(cam, self.engine)
            for cam in CAMS
        }

    def image_callback(self, msg):
        cam_name = msg.header.frame_id
        T_WC = self.pose.get_T_world_cam(cam_name, msg.header.stamp)

        if not self.keyframe.should_accept(T_WC, msg, detections):
            return

        world = self.pipelines[cam_name].process_frame(
            msg.image,
            msg.detections,
            T_WC
        )

        self.publish(world)
