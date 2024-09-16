"""Blender script used to generate the synthetic dataset.
reference:  https://github.com/georg-wolflein/chesscog/blob/master/scripts/synthesize_data.py
"""

import bpy
import bpy_extras.object_utils
import mathutils
import numpy as np
from pathlib import Path
import typing
import json
import sys

import os 
import shutil

import random

MIN_BOARD_CORNER_PADDING = 25  # pixels
SQUARE_LENGTH = 0.036
# units (1 unit is the side length of a chessboard square)
CAMERA_DISTANCE = 11
MAX_OBJ_CNT = 3

# rotation = 2pi/ROTATION_LIMIT
ROTATION_LIMIT = 1.5
# location = random(-MAX_LOCATION_OFFSET, MAX_LOCATION_OFFSET)
MAX_LOCATION_OFFSET = 3
# scale = random(MIN_SCALE, 1)
MIN_SCALE = 0.25
MAX_SCALE = 3

#% an image is obstructed by an object
OBSTRUCTION_RATE = 0.5

def point_to(obj, focus: mathutils.Vector, roll: float = 0):
    # Based on https://blender.stackexchange.com/a/127440
    loc = obj.location
    direction = focus - loc
    quat = direction.to_track_quat("-Z", "Y").to_matrix().to_4x4()
    roll_matrix = mathutils.Matrix.Rotation(roll, 4, "Z")
    loc = loc.to_tuple()
    obj.matrix_world = quat @ roll_matrix
    obj.location = loc


def setup_spotlight(light) -> dict:
    angle_xy_plane = np.random.randint(0, 360)
    focus_x, focus_y = np.random.multivariate_normal(
        (0., 0.), np.eye(2) * 2.5 * SQUARE_LENGTH)
    focus = mathutils.Vector((focus_x, focus_y, 0.))
    radius = 10 * SQUARE_LENGTH
    x = radius * np.cos(np.deg2rad(angle_xy_plane))
    y = radius * np.sin(np.deg2rad(angle_xy_plane))
    z = np.random.uniform(5, 10) * SQUARE_LENGTH
    location = mathutils.Vector((x, y, z))
    light.location = location
    point_to(light, focus)
    return {
        "xy_angle": angle_xy_plane,
        "focus": focus.to_tuple(),
        "location": location.to_tuple()
    }


def setup_lighting() -> dict:
    flash = bpy.data.objects["Camera flash light"]
    spot1 = bpy.data.objects["Spot 1"]
    spot2 = bpy.data.objects["Spot 2"]

    modes = {
        "flash": {
            flash: True,
            spot1: False,
            spot2: False
        },
        "spotlights": {
            flash: False,
            spot1: True,
            spot2: True
        }
    }
    mode, visibilities = list(modes.items())[np.random.randint(len(modes))]

    for obj, visibility in visibilities.items():
        obj.hide_render = not visibility

    return {
        "mode": mode,
        "flash": {
            "active": not flash.hide_render
        },
        **{
            key: {
                "active": not obj.hide_render,
                **setup_spotlight(obj)
            } for (key, obj) in {"spot1": spot1, "spot2": spot2}.items()
        }
    }

def init_object(collection, pass_index):
    name = "Window"

    src_obj = bpy.data.objects[name]
    obj = src_obj.copy()
    obj.data = src_obj.data.copy()
    obj.animation_data_clear()

    MAX_LOCATION_OFFSET_2 = MAX_LOCATION_OFFSET * 2

    location = (np.random.rand() * MAX_LOCATION_OFFSET_2 - MAX_LOCATION_OFFSET, np.random.rand() * MAX_LOCATION_OFFSET_2 - MAX_LOCATION_OFFSET, np.random.rand() * MAX_LOCATION_OFFSET_2 - MAX_LOCATION_OFFSET)
    rotation = (np.random.rand() * 3.14/ROTATION_LIMIT, np.random.rand() * 3.14/ROTATION_LIMIT, np.random.rand() * 3.14/ROTATION_LIMIT)

    scale_val = np.random.rand() * (MAX_SCALE - MIN_SCALE) + MIN_SCALE
    obj.scale = (scale_val, scale_val, scale_val)

    obj.location = location

    #add pass index for object instance labelling
    obj.pass_index = pass_index
    
    # obj.rotation_mode = "XYZ"
    obj.rotation_euler = rotation
    # obj.rotation_mode = prev_mode
    collection.objects.link(obj)
    return obj

# rotation = pi/ROTATION_LIMIT
OBS_ROTATION_LIMIT = 3
# location = random(-MAX_LOCATION_OFFSET, MAX_LOCATION_OFFSET)
OBS_MAX_LOCATION_OFFSET = 1.5
OBS_MIN_SCALE = 1.5
OBS_MAX_SCALE = 2

def obstruction(obstruction_objs, active_collection):
    # pick a random object
    src_obj = random.choice(obstruction_objs.objects)

    obj = src_obj.copy()
    obj.data = src_obj.data.copy()
    obj.animation_data_clear()

    MAX_LOCATION_OFFSET_2 = OBS_MAX_LOCATION_OFFSET * 2

    location = (np.random.rand() * MAX_LOCATION_OFFSET_2 - OBS_MAX_LOCATION_OFFSET, np.random.rand() * MAX_LOCATION_OFFSET_2 - OBS_MAX_LOCATION_OFFSET, np.random.rand() * MAX_LOCATION_OFFSET_2 - OBS_MAX_LOCATION_OFFSET)
    rotation = (np.random.rand() * 3.14/OBS_ROTATION_LIMIT, np.random.rand() * 3.14/OBS_ROTATION_LIMIT, np.random.rand() * 3.14/OBS_ROTATION_LIMIT)

    scale_range = OBS_MAX_SCALE - OBS_MIN_SCALE
    scale_val = np.random.rand() * (scale_range) + OBS_MIN_SCALE
    obj.scale = (scale_val, scale_val, scale_val)

    obj.location = location
    
    obj.rotation_euler = rotation
    active_collection.objects.link(obj)
    return obj

COLLECTION_NAME = "Windows"

MAX_PASS_INDEX = 1000
def render(output_file: Path, output_mask_file: Path, background_file: Path, obstruction_flag: bool):
    scene = bpy.context.scene

    scene.use_nodes = True
    scene.node_tree.nodes.clear()

    # Create the render layer node
    render_node = scene.node_tree.nodes.new("CompositorNodeRLayers")


    # mask outputting
    output_mask_node = scene.node_tree.nodes.new("CompositorNodeOutputFile")
    output_mask_node.base_path = str(output_mask_file)

    output_mask_node.format.file_format = "PNG"

    divide_node = scene.node_tree.nodes.new("CompositorNodeMath")
    divide_node.operation = "DIVIDE"

    scene.node_tree.links.new(render_node.outputs["IndexOB"], divide_node.inputs[0])
    divide_node.inputs[1].default_value = MAX_PASS_INDEX
    scene.node_tree.links.new(divide_node.outputs["Value"], output_mask_node.inputs["Image"])

    # Add background for image node
    bg_raw_node = scene.node_tree.nodes.new("CompositorNodeImage")
    bg_raw_node.image = bpy.data.images.load(background_file)
    bg_node = scene.node_tree.nodes.new("CompositorNodeScale")
    bg_node.space = "RENDER_SIZE"
    bg_node.frame_method = "STRETCH"
    scene.node_tree.links.new(bg_raw_node.outputs["Image"], bg_node.inputs["Image"])

    mix_node = scene.node_tree.nodes.new("CompositorNodeMixRGB")
    mix_node.blend_type = "MIX"
    mix_node.use_alpha = True

    scene.node_tree.links.new(bg_node.outputs["Image"], mix_node.inputs[1])
    scene.node_tree.links.new(render_node.outputs["Image"], mix_node.inputs[2])

    output_rgb_node = scene.node_tree.nodes.new("CompositorNodeOutputFile")
    output_rgb_node.base_path = str(output_file)
    output_rgb_node.format.file_format = "PNG"
    scene.node_tree.links.new(mix_node.outputs["Image"], output_rgb_node.inputs["Image"])

    # Create a collection to store the position
    if COLLECTION_NAME not in bpy.data.collections:
        collection = bpy.data.collections.new(COLLECTION_NAME)
        scene.collection.children.link(collection)
    collection = bpy.data.collections[COLLECTION_NAME]


    with bpy.context.temp_override(selected_objects=collection.objects):

    # Remove all objects from the collection
        bpy.ops.object.delete()

    obj_count = np.random.randint(1, MAX_OBJ_CNT+1)

    for i in range(obj_count):
        init_object(collection, pass_index = int((MAX_PASS_INDEX//MAX_OBJ_CNT) * (MAX_OBJ_CNT - i)))

    if obstruction_flag:
        obstruction(bpy.data.collections["Obstructions"], collection)

    # Perform the rendering
    bpy.ops.render.render(write_still=1)

def init_scene():
    scene = bpy.context.scene

    # Setup rendering
    scene.render.engine = "CYCLES"
    scene.cycles.sample = 4
    scene.cycles.adaptive_threshold = 0.1
    scene.render.image_settings.file_format = "JPEG"
    scene.render.resolution_x = 640
    scene.render.resolution_y = 360

    scene.render.use_persistent_data = True

    bpy.context.preferences.addons[
    "cycles"
    ].preferences.compute_device_type = "OPTIX" # or "OPENCL"

    scene.cycles.device = "GPU"


STARTING_COUNT = 0
#amount of current images
ENDING_COUNT = 4087
if __name__ == "__main__":

    dir_path = os.path.dirname(os.path.realpath(__file__)).replace("dataset.blend", "")

    # camera_params = setup_camera()
    # lighting_params = setup_lighting()

    init_scene()

    background_dir = f"{dir_path}/backgrounds"
    backgrounds = [os.path.join(background_dir, f) for f in os.listdir(background_dir) if os.path.isfile(os.path.join(background_dir, f))]

    for i in range(STARTING_COUNT, ENDING_COUNT):
        output_file = Path(f"{dir_path}/output/output_{i}")
        output_mask_file = Path(f"{dir_path}/output/instance/output_{i}")

        this_background = random.choice(backgrounds)

        rand_obstruction = np.random.rand()
        obstruction_flag = rand_obstruction < OBSTRUCTION_RATE

        render(output_file, output_mask_file, this_background, obstruction_flag)

    #because using the file output node, the image is save in seperate folders.
    #we need to move the images to the output folder
    src = f"{dir_path}/output/instance"
    src_file = src + "/Image0001.png"
    if os.path.isfile(src_file):
        os.remove(src_file)

    src = f"{dir_path}/output"
    src_file = src + "/Image0001.png"
    if os.path.isfile(src_file):
        os.remove(src_file)


    for i in range(STARTING_COUNT, ENDING_COUNT):
        src = f"{dir_path}/output/instance/output_{i}"
        src_files = [os.path.join(src, f) for f in os.listdir(src) if os.path.isfile(os.path.join(src, f))]
        src_file = src_files[0]
        dst = f"{dir_path}/output/instance/output_{i}.png"
        shutil.move(Path(src_file), Path(dst))
        shutil.rmtree(Path(src))
        
        src = f"{dir_path}/output/output_{i}"
        src_files = [os.path.join(src, f) for f in os.listdir(src) if os.path.isfile(os.path.join(src, f))]
        src_file = src_files[0]
        dst = f"{dir_path}/output/output_{i}.png"
        shutil.move(Path(src_file), Path(dst))
        shutil.rmtree(Path(src))


