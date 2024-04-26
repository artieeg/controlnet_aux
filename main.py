from PIL import Image

from src.controlnet_aux import OpenposeDetector
openpose = OpenposeDetector.from_pretrained("lllyasviel/Annotators")

pose = Image.open("./img1.png")
pose_2 = Image.open("./img2.png")

map, poses = openpose(pose)
print(type(poses[0]))

freeze_pose_list = [[
    'left_leg'
]]

new_map, _ = openpose(pose_2, freeze_poses_idx=[0], freeze_poses=[
                      poses[0]], freeze_parts=["left_leg", "right_leg"])

map.show()
new_map.show()
