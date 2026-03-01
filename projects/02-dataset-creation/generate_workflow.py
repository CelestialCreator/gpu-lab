#!/usr/bin/env python3
"""Generate ComfyUI workflow JSON for Character LoRA Dataset Creation Pipeline.

Produces a single workflow with:
- Part 1: 18 generation groups using Qwen Image Edit for character variations
- Part 2: Dataset preparation with Florence2 captioning, upscaling, and paired output
"""

import json
import random
import uuid


class WorkflowBuilder:
    def __init__(self):
        self.nodes = []
        self.links = []
        self.groups = []
        self._next_node_id = 1
        self._next_link_id = 1

    def _nid(self):
        nid = self._next_node_id
        self._next_node_id += 1
        return nid

    def _lid(self):
        lid = self._next_link_id
        self._next_link_id += 1
        return lid

    def _get_node(self, node_id):
        for n in self.nodes:
            if n["id"] == node_id:
                return n
        raise ValueError(f"Node {node_id} not found")

    def add_node(self, node_type, pos, size, widgets_values=None,
                 title=None, properties=None, flags=None, mode=0, color=None, bgcolor=None):
        nid = self._nid()
        node = {
            "id": nid,
            "type": node_type,
            "pos": pos,
            "size": size,
            "flags": flags or {},
            "order": 0,
            "mode": mode,
            "inputs": [],
            "outputs": [],
            "properties": properties or {},
            "widgets_values": widgets_values or [],
            "shape": 1,
        }
        if title:
            node["title"] = title
        if color:
            node["color"] = color
        if bgcolor:
            node["bgcolor"] = bgcolor
        self.nodes.append(node)
        return nid

    def add_output(self, node_id, name, type_name):
        node = self._get_node(node_id)
        output = {"name": name, "type": type_name, "links": []}
        node["outputs"].append(output)
        return len(node["outputs"]) - 1

    def add_input(self, node_id, name, type_name, widget=None, shape=None):
        node = self._get_node(node_id)
        inp = {"name": name, "type": type_name, "link": None}
        if widget:
            inp["widget"] = widget
        if shape:
            inp["shape"] = shape
        node["inputs"].append(inp)
        return len(node["inputs"]) - 1

    def connect(self, src_id, src_slot, dst_id, dst_slot, type_name):
        link_id = self._lid()
        self.links.append([link_id, src_id, src_slot, dst_id, dst_slot, type_name])
        src_node = self._get_node(src_id)
        src_node["outputs"][src_slot]["links"].append(link_id)
        dst_node = self._get_node(dst_id)
        dst_node["inputs"][dst_slot]["link"] = link_id
        return link_id

    def add_group(self, title, bounding, color="#7155be"):
        self.groups.append({
            "id": len(self.groups) + 1,
            "title": title,
            "bounding": bounding,
            "color": color,
            "font_size": 24,
            "flags": {}
        })

    def build(self):
        for i, node in enumerate(self.nodes):
            node["order"] = i
        return {
            "id": str(uuid.uuid4()),
            "revision": 0,
            "last_node_id": self._next_node_id - 1,
            "last_link_id": self._next_link_id - 1,
            "nodes": self.nodes,
            "links": self.links,
            "groups": self.groups,
            "config": {},
            "extra": {
                "ds": {"scale": 0.25, "offset": [200, 200]},
                "groupNodes": {}
            },
            "version": 0.4
        }


def make_set_node(wb, name, type_name, pos, color=None, bgcolor=None):
    """Create a SetNode for sharing values across groups."""
    nid = wb.add_node("SetNode", pos, [210, 60],
                       widgets_values=[name],
                       title=f"Set_{name}",
                       properties={"previousName": name},
                       flags={"collapsed": True},
                       color=color, bgcolor=bgcolor)
    wb.add_input(nid, type_name, type_name)
    wb.add_output(nid, "*", "*")
    return nid


def make_get_node(wb, name, type_name, pos, color=None, bgcolor=None):
    """Create a GetNode for retrieving shared values."""
    nid = wb.add_node("GetNode", pos, [210, 58],
                       widgets_values=[name],
                       title=f"Get_{name}",
                       properties={},
                       flags={"collapsed": True},
                       color=color, bgcolor=bgcolor)
    wb.add_output(nid, type_name, type_name)
    return nid


def build_shared_infrastructure(wb):
    """Build shared model loading and parameter nodes. Returns dict of node IDs."""
    ids = {}

    # --- GGUF Model Loader ---
    ids["gguf_loader"] = wb.add_node(
        "LoaderGGUF", [100, 100], [360, 70],
        widgets_values=["gguf/Qwen-Image-Edit-2509-Q5_0.gguf"],
        properties={"cnr_id": "gguf", "Node name for S&R": "LoaderGGUF"})
    wb.add_output(ids["gguf_loader"], "MODEL", "MODEL")

    # --- ModelSamplingAuraFlow (shift=3) ---
    ids["aura_flow"] = wb.add_node(
        "ModelSamplingAuraFlow", [100, 200], [360, 60],
        widgets_values=[3],
        properties={"cnr_id": "comfy-core", "Node name for S&R": "ModelSamplingAuraFlow"},
        flags={"collapsed": True})
    wb.add_input(ids["aura_flow"], "model", "MODEL")
    wb.add_output(ids["aura_flow"], "MODEL", "MODEL")
    wb.connect(ids["gguf_loader"], 0, ids["aura_flow"], 0, "MODEL")

    # --- CFGNorm (cfg=1) ---
    ids["cfg_norm"] = wb.add_node(
        "CFGNorm", [100, 240], [360, 60],
        widgets_values=[1],
        properties={"cnr_id": "comfy-core", "Node name for S&R": "CFGNorm"},
        flags={"collapsed": True})
    wb.add_input(ids["cfg_norm"], "model", "MODEL")
    wb.add_output(ids["cfg_norm"], "patched_model", "MODEL")
    wb.connect(ids["aura_flow"], 0, ids["cfg_norm"], 0, "MODEL")

    # --- LoRA Loader (Lightning 4-steps) ---
    ids["lora_loader"] = wb.add_node(
        "LoraLoaderModelOnly", [100, 280], [360, 90],
        widgets_values=["qwen/Qwen-Image-Lightning-4steps-V2.0.safetensors", 1],
        properties={"cnr_id": "comfy-core", "Node name for S&R": "LoraLoaderModelOnly"})
    wb.add_input(ids["lora_loader"], "model", "MODEL")
    wb.add_output(ids["lora_loader"], "MODEL", "MODEL")
    wb.connect(ids["cfg_norm"], 0, ids["lora_loader"], 0, "MODEL")

    # --- Set_model ---
    ids["set_model"] = make_set_node(wb, "model", "MODEL", [100, 400],
                                      color="#223", bgcolor="#335")
    wb.connect(ids["lora_loader"], 0, ids["set_model"], 0, "MODEL")

    # --- CLIP Loader ---
    ids["clip_loader"] = wb.add_node(
        "CLIPLoader", [100, 460], [350, 110],
        widgets_values=["qwen/qwen_2.5_vl_7b_fp8_scaled.safetensors", "qwen_image", "cpu"],
        properties={"cnr_id": "comfy-core", "Node name for S&R": "CLIPLoader"})
    wb.add_output(ids["clip_loader"], "CLIP", "CLIP")

    # --- Set_clip ---
    ids["set_clip"] = make_set_node(wb, "clip", "CLIP", [100, 600],
                                     color="#432", bgcolor="#653")
    wb.connect(ids["clip_loader"], 0, ids["set_clip"], 0, "CLIP")

    # --- VAE Loader ---
    ids["vae_loader"] = wb.add_node(
        "VAELoader", [100, 650], [350, 60],
        widgets_values=["qwen/qwen_image_vae.safetensors"],
        properties={"cnr_id": "comfy-core", "Node name for S&R": "VAELoader"})
    wb.add_output(ids["vae_loader"], "VAE", "VAE")

    # --- Set_vae ---
    ids["set_vae"] = make_set_node(wb, "vae", "VAE", [100, 740],
                                    color="#322", bgcolor="#533")
    wb.connect(ids["vae_loader"], 0, ids["set_vae"], 0, "VAE")

    # --- PrimitiveInt STEPS (4) ---
    ids["steps_prim"] = wb.add_node(
        "PrimitiveInt", [100, 800], [210, 82],
        widgets_values=[4, "fixed"],
        title="STEPS",
        properties={"cnr_id": "comfy-core", "Node name for S&R": "PrimitiveInt"})
    wb.add_output(ids["steps_prim"], "INT", "INT")

    ids["set_steps"] = make_set_node(wb, "steps", "INT", [100, 910])
    wb.connect(ids["steps_prim"], 0, ids["set_steps"], 0, "INT")

    # --- PrimitiveFloat CFG (1) ---
    ids["cfg_prim"] = wb.add_node(
        "PrimitiveFloat", [320, 800], [210, 80],
        widgets_values=[1],
        title="CFG",
        properties={"cnr_id": "comfy-core", "Node name for S&R": "PrimitiveFloat"})
    wb.add_output(ids["cfg_prim"], "FLOAT", "FLOAT")

    ids["set_cfg"] = make_set_node(wb, "cfg", "FLOAT", [320, 910],
                                    color="#232", bgcolor="#353")
    wb.connect(ids["cfg_prim"], 0, ids["set_cfg"], 0, "FLOAT")

    # --- PrimitiveInt WIDTH (1024) ---
    ids["width_prim"] = wb.add_node(
        "PrimitiveInt", [100, 970], [210, 82],
        widgets_values=[1024, "fixed"],
        title="WIDTH",
        properties={"cnr_id": "comfy-core", "Node name for S&R": "PrimitiveInt"})
    wb.add_output(ids["width_prim"], "INT", "INT")

    ids["set_width"] = make_set_node(wb, "Width", "INT", [100, 1080])
    wb.connect(ids["width_prim"], 0, ids["set_width"], 0, "INT")

    # --- PrimitiveInt HEIGHT (1024) ---
    ids["height_prim"] = wb.add_node(
        "PrimitiveInt", [320, 970], [210, 82],
        widgets_values=[1024, "fixed"],
        title="HEIGHT",
        properties={"cnr_id": "comfy-core", "Node name for S&R": "PrimitiveInt"})
    wb.add_output(ids["height_prim"], "INT", "INT")

    ids["set_height"] = make_set_node(wb, "Height", "INT", [320, 1080])
    wb.connect(ids["height_prim"], 0, ids["set_height"], 0, "INT")

    # --- Load Image (main input photo) ---
    ids["load_image_main"] = wb.add_node(
        "LoadImage", [100, 1160], [420, 460],
        widgets_values=["example.png", "image"],
        title="INPUT PHOTO",
        properties={"cnr_id": "comfy-core", "Node name for S&R": "LoadImage"})
    wb.add_output(ids["load_image_main"], "IMAGE", "IMAGE")
    wb.add_output(ids["load_image_main"], "MASK", "MASK")

    # Scale input to 1 megapixel
    ids["scale_main"] = wb.add_node(
        "ImageScaleToTotalPixels", [100, 1660], [270, 82],
        widgets_values=["lanczos", 1],
        properties={"cnr_id": "comfy-core", "Node name for S&R": "ImageScaleToTotalPixels"},
        flags={"collapsed": True})
    wb.add_input(ids["scale_main"], "image", "IMAGE")
    wb.add_output(ids["scale_main"], "IMAGE", "IMAGE")
    wb.connect(ids["load_image_main"], 0, ids["scale_main"], 0, "IMAGE")

    ids["set_input_image"] = make_set_node(wb, "input_image", "IMAGE", [100, 1710])
    wb.connect(ids["scale_main"], 0, ids["set_input_image"], 0, "IMAGE")

    # --- Load Image (clothing reference for virtual try-on) ---
    ids["load_image_clothing"] = wb.add_node(
        "LoadImage", [100, 1780], [420, 460],
        widgets_values=["clothing_ref.png", "image"],
        title="CLOTHING REF (Groups 15-16)",
        properties={"cnr_id": "comfy-core", "Node name for S&R": "LoadImage"})
    wb.add_output(ids["load_image_clothing"], "IMAGE", "IMAGE")
    wb.add_output(ids["load_image_clothing"], "MASK", "MASK")

    ids["scale_clothing"] = wb.add_node(
        "ImageScaleToTotalPixels", [100, 2280], [270, 82],
        widgets_values=["lanczos", 1],
        properties={"cnr_id": "comfy-core", "Node name for S&R": "ImageScaleToTotalPixels"},
        flags={"collapsed": True})
    wb.add_input(ids["scale_clothing"], "image", "IMAGE")
    wb.add_output(ids["scale_clothing"], "IMAGE", "IMAGE")
    wb.connect(ids["load_image_clothing"], 0, ids["scale_clothing"], 0, "IMAGE")

    ids["set_clothing_ref"] = make_set_node(wb, "clothing_ref", "IMAGE", [100, 2330])
    wb.connect(ids["scale_clothing"], 0, ids["set_clothing_ref"], 0, "IMAGE")

    # --- Load Image (pose reference for pose transfer) ---
    ids["load_image_pose"] = wb.add_node(
        "LoadImage", [100, 2400], [420, 460],
        widgets_values=["pose_ref.png", "image"],
        title="POSE REF (Groups 17-18)",
        properties={"cnr_id": "comfy-core", "Node name for S&R": "LoadImage"})
    wb.add_output(ids["load_image_pose"], "IMAGE", "IMAGE")
    wb.add_output(ids["load_image_pose"], "MASK", "MASK")

    ids["scale_pose"] = wb.add_node(
        "ImageScaleToTotalPixels", [100, 2900], [270, 82],
        widgets_values=["lanczos", 1],
        properties={"cnr_id": "comfy-core", "Node name for S&R": "ImageScaleToTotalPixels"},
        flags={"collapsed": True})
    wb.add_input(ids["scale_pose"], "image", "IMAGE")
    wb.add_output(ids["scale_pose"], "IMAGE", "IMAGE")
    wb.connect(ids["load_image_pose"], 0, ids["scale_pose"], 0, "IMAGE")

    ids["set_pose_ref"] = make_set_node(wb, "pose_ref", "IMAGE", [100, 2950])
    wb.connect(ids["scale_pose"], 0, ids["set_pose_ref"], 0, "IMAGE")

    # --- Trigger Word (StringConstant from KJNodes) ---
    ids["trigger_word"] = wb.add_node(
        "StringConstant", [100, 3030], [300, 60],
        widgets_values=["my_character"],
        title="TRIGGER WORD",
        properties={"cnr_id": "comfyui-kjnodes", "Node name for S&R": "StringConstant"})
    wb.add_output(ids["trigger_word"], "STRING", "STRING")

    ids["set_trigger"] = make_set_node(wb, "trigger_word", "STRING", [100, 3120])
    wb.connect(ids["trigger_word"], 0, ids["set_trigger"], 0, "STRING")

    return ids


def build_generation_group(wb, group_num, group_name, prompt, x_offset,
                           image_source="input_image", image2_source=None,
                           save_prefix="character", extra_set_name=None):
    """Build a single generation group. Returns dict of node IDs."""
    ids = {}
    y_base = 100
    x = x_offset

    # Get nodes for shared resources
    ids["get_clip"] = make_get_node(wb, "clip", "CLIP", [x, y_base],
                                     color="#432", bgcolor="#653")
    ids["get_vae"] = make_get_node(wb, "vae", "VAE", [x, y_base + 40],
                                    color="#322", bgcolor="#533")
    ids["get_model"] = make_get_node(wb, "model", "MODEL", [x + 220, y_base + 280],
                                      color="#223", bgcolor="#335")
    ids["get_steps"] = make_get_node(wb, "steps", "INT", [x + 220, y_base + 320])
    ids["get_cfg"] = make_get_node(wb, "cfg", "FLOAT", [x + 220, y_base + 360],
                                    color="#232", bgcolor="#353")
    ids["get_width"] = make_get_node(wb, "Width", "INT", [x, y_base + 400])
    ids["get_height"] = make_get_node(wb, "Height", "INT", [x, y_base + 440])

    # Get input image
    ids["get_image"] = make_get_node(wb, image_source, "IMAGE", [x, y_base + 80])

    # Optional: Get image2 for try-on/pose transfer
    if image2_source:
        ids["get_image2"] = make_get_node(wb, image2_source, "IMAGE", [x, y_base + 120])

    # TextEncodeQwenImageEditPlus
    ids["text_encode"] = wb.add_node(
        "TextEncodeQwenImageEditPlus", [x, y_base + 160], [440, 280],
        widgets_values=[prompt],
        properties={"cnr_id": "comfy-core", "Node name for S&R": "TextEncodeQwenImageEditPlus"})
    wb.add_input(ids["text_encode"], "clip", "CLIP")
    wb.add_input(ids["text_encode"], "vae", "VAE", shape=7)
    wb.add_input(ids["text_encode"], "image1", "IMAGE", shape=7)
    wb.add_input(ids["text_encode"], "image2", "IMAGE", shape=7)
    wb.add_input(ids["text_encode"], "image3", "IMAGE", shape=7)
    wb.add_output(ids["text_encode"], "CONDITIONING", "CONDITIONING")

    # Wire clip, vae, image1
    wb.connect(ids["get_clip"], 0, ids["text_encode"], 0, "CLIP")
    wb.connect(ids["get_vae"], 0, ids["text_encode"], 1, "VAE")
    wb.connect(ids["get_image"], 0, ids["text_encode"], 2, "IMAGE")

    # Wire image2 if present
    if image2_source:
        wb.connect(ids["get_image2"], 0, ids["text_encode"], 3, "IMAGE")

    # EmptySD3LatentImage
    ids["empty_latent"] = wb.add_node(
        "EmptySD3LatentImage", [x, y_base + 480], [390, 120],
        widgets_values=[1024, 1024, 1],
        properties={"cnr_id": "comfy-core", "Node name for S&R": "EmptySD3LatentImage"},
        flags={"collapsed": True})
    wb.add_input(ids["empty_latent"], "width", "INT", widget={"name": "width"})
    wb.add_input(ids["empty_latent"], "height", "INT", widget={"name": "height"})
    wb.add_output(ids["empty_latent"], "LATENT", "LATENT")
    wb.connect(ids["get_width"], 0, ids["empty_latent"], 0, "INT")
    wb.connect(ids["get_height"], 0, ids["empty_latent"], 1, "INT")

    # KSampler
    seed = random.randint(1, 2**53)
    ids["ksampler"] = wb.add_node(
        "KSampler", [x + 460, y_base + 160], [390, 484],
        widgets_values=[seed, "fixed", 4, 1, "euler", "simple", 1],
        properties={"cnr_id": "comfy-core", "Node name for S&R": "KSampler"})
    wb.add_input(ids["ksampler"], "model", "MODEL")
    wb.add_input(ids["ksampler"], "positive", "CONDITIONING")
    wb.add_input(ids["ksampler"], "negative", "CONDITIONING")
    wb.add_input(ids["ksampler"], "latent_image", "LATENT")
    wb.add_input(ids["ksampler"], "steps", "INT", widget={"name": "steps"})
    wb.add_input(ids["ksampler"], "cfg", "FLOAT", widget={"name": "cfg"})
    wb.add_output(ids["ksampler"], "LATENT", "LATENT")

    # Wire KSampler
    wb.connect(ids["get_model"], 0, ids["ksampler"], 0, "MODEL")
    wb.connect(ids["text_encode"], 0, ids["ksampler"], 1, "CONDITIONING")  # positive
    wb.connect(ids["text_encode"], 0, ids["ksampler"], 2, "CONDITIONING")  # negative (same)
    wb.connect(ids["empty_latent"], 0, ids["ksampler"], 3, "LATENT")
    wb.connect(ids["get_steps"], 0, ids["ksampler"], 4, "INT")
    wb.connect(ids["get_cfg"], 0, ids["ksampler"], 5, "FLOAT")

    # Get VAE for decode (reuse the existing get_vae - need a second one)
    ids["get_vae2"] = make_get_node(wb, "vae", "VAE", [x + 460, y_base + 660],
                                     color="#322", bgcolor="#533")

    # VAEDecode
    ids["vae_decode"] = wb.add_node(
        "VAEDecode", [x + 700, y_base + 660], [190, 46],
        properties={"cnr_id": "comfy-core", "Node name for S&R": "VAEDecode"},
        flags={"collapsed": True})
    wb.add_input(ids["vae_decode"], "samples", "LATENT")
    wb.add_input(ids["vae_decode"], "vae", "VAE")
    wb.add_output(ids["vae_decode"], "IMAGE", "IMAGE")
    wb.connect(ids["ksampler"], 0, ids["vae_decode"], 0, "LATENT")
    wb.connect(ids["get_vae2"], 0, ids["vae_decode"], 1, "VAE")

    # SaveImage — positioned below the KSampler
    ids["save_image"] = wb.add_node(
        "SaveImage", [x + 460, y_base + 730], [500, 460],
        widgets_values=[f"{save_prefix}/{group_name}"],
        properties={"cnr_id": "comfy-core", "Node name for S&R": "SaveImage"})
    wb.add_input(ids["save_image"], "images", "IMAGE")
    wb.connect(ids["vae_decode"], 0, ids["save_image"], 0, "IMAGE")

    # Optional: SetNode for this group's output (used when other groups reference it)
    if extra_set_name:
        ids["set_output"] = make_set_node(wb, extra_set_name, "IMAGE",
                                           [x + 700, y_base + 720])
        wb.connect(ids["vae_decode"], 0, ids["set_output"], 0, "IMAGE")

    return ids


def build_part2_dataset_prep(wb, x_offset, y_offset):
    """Build Part 2: Dataset preparation pipeline.
    All Part 2 nodes start muted (mode=2) since they run separately."""
    ids = {}
    x = x_offset
    y = y_offset
    mode = 2  # muted by default

    # --- LoadImagesFromFolderKJ ---
    ids["load_folder"] = wb.add_node(
        "LoadImagesFromFolderKJ", [x, y], [400, 200],
        widgets_values=["output/my_character", 1024, 1024, "crop", 0, 0, False],
        title="Load Generated Images",
        properties={"cnr_id": "comfyui-kjnodes", "Node name for S&R": "LoadImagesFromFolderKJ"},
        mode=mode)
    wb.add_output(ids["load_folder"], "IMAGE", "IMAGE")
    wb.add_output(ids["load_folder"], "MASK", "MASK")
    wb.add_output(ids["load_folder"], "INT", "INT")
    wb.add_output(ids["load_folder"], "STRING", "STRING")

    # --- Florence2 Model Loader ---
    ids["fl2_loader"] = wb.add_node(
        "DownloadAndLoadFlorence2Model", [x, y + 250], [400, 120],
        widgets_values=["microsoft/Florence-2-base", "fp16", "sdpa", False],
        title="Load Florence2",
        properties={"Node name for S&R": "DownloadAndLoadFlorence2Model"},
        mode=mode)
    wb.add_output(ids["fl2_loader"], "FL2MODEL", "FL2MODEL")

    # --- Florence2Run (captioning) ---
    ids["fl2_run"] = wb.add_node(
        "Florence2Run", [x + 450, y], [440, 350],
        widgets_values=["", "detailed_caption", True, False, 1024, 3, True, "", 1],
        title="Auto-Caption Images",
        properties={"Node name for S&R": "Florence2Run"},
        mode=mode)
    wb.add_input(ids["fl2_run"], "image", "IMAGE")
    wb.add_input(ids["fl2_run"], "florence2_model", "FL2MODEL")
    wb.add_input(ids["fl2_run"], "text_input", "STRING")
    wb.add_output(ids["fl2_run"], "IMAGE", "IMAGE")
    wb.add_output(ids["fl2_run"], "MASK", "MASK")
    wb.add_output(ids["fl2_run"], "STRING", "STRING")
    wb.add_output(ids["fl2_run"], "JSON", "JSON")

    wb.connect(ids["load_folder"], 0, ids["fl2_run"], 0, "IMAGE")
    wb.connect(ids["fl2_loader"], 0, ids["fl2_run"], 1, "FL2MODEL")

    # --- Get trigger word ---
    ids["get_trigger"] = make_get_node(wb, "trigger_word", "STRING", [x + 450, y + 370])

    # --- JoinStrings (prepend trigger word) ---
    ids["join_strings"] = wb.add_node(
        "JoinStrings", [x + 450, y + 410], [300, 100],
        widgets_values=[", "],
        title="Prepend Trigger Word",
        properties={"cnr_id": "comfyui-kjnodes", "Node name for S&R": "JoinStrings"},
        mode=mode)
    wb.add_input(ids["join_strings"], "delimiter", "STRING")
    wb.add_input(ids["join_strings"], "string1", "STRING", shape=7)
    wb.add_input(ids["join_strings"], "string2", "STRING", shape=7)
    wb.add_output(ids["join_strings"], "STRING", "STRING")

    wb.connect(ids["get_trigger"], 0, ids["join_strings"], 1, "STRING")
    wb.connect(ids["fl2_run"], 2, ids["join_strings"], 2, "STRING")  # caption output

    # --- Upscale Model Loader ---
    ids["upscale_loader"] = wb.add_node(
        "UpscaleModelLoader", [x + 920, y], [350, 60],
        widgets_values=["4x-UltraSharp.pth"],
        title="Load 4x-UltraSharp",
        properties={"cnr_id": "comfy-core", "Node name for S&R": "UpscaleModelLoader"},
        mode=mode)
    wb.add_output(ids["upscale_loader"], "UPSCALE_MODEL", "UPSCALE_MODEL")

    # --- ImageUpscaleWithModel ---
    ids["upscale"] = wb.add_node(
        "ImageUpscaleWithModel", [x + 920, y + 100], [350, 80],
        title="4x Upscale",
        properties={"cnr_id": "comfy-core", "Node name for S&R": "ImageUpscaleWithModel"},
        mode=mode)
    wb.add_input(ids["upscale"], "upscale_model", "UPSCALE_MODEL")
    wb.add_input(ids["upscale"], "image", "IMAGE")
    wb.add_output(ids["upscale"], "IMAGE", "IMAGE")

    wb.connect(ids["upscale_loader"], 0, ids["upscale"], 0, "UPSCALE_MODEL")
    wb.connect(ids["load_folder"], 0, ids["upscale"], 1, "IMAGE")

    # --- ImageResizeKJv2 (resize to 1024x1024) ---
    ids["resize"] = wb.add_node(
        "ImageResizeKJv2", [x + 920, y + 220], [350, 336],
        widgets_values=[1024, 1024, "lanczos", "resize", "0, 0, 0", "center", 2],
        title="Resize to 1024px",
        properties={"cnr_id": "comfyui-kjnodes", "Node name for S&R": "ImageResizeKJv2"},
        mode=mode)
    wb.add_input(ids["resize"], "image", "IMAGE")
    wb.add_input(ids["resize"], "mask", "MASK", shape=7)
    wb.add_output(ids["resize"], "IMAGE", "IMAGE")
    wb.add_output(ids["resize"], "INT", "INT")
    wb.add_output(ids["resize"], "INT", "INT")
    wb.add_output(ids["resize"], "MASK", "MASK")

    wb.connect(ids["upscale"], 0, ids["resize"], 0, "IMAGE")

    # --- SaveImage (save .png files) ---
    ids["save_images"] = wb.add_node(
        "SaveImage", [x + 1320, y], [500, 460],
        widgets_values=["my_character_dataset/img"],
        title="Save Dataset Images",
        properties={"cnr_id": "comfy-core", "Node name for S&R": "SaveImage"},
        mode=mode)
    wb.add_input(ids["save_images"], "images", "IMAGE")
    wb.connect(ids["resize"], 0, ids["save_images"], 0, "IMAGE")

    # --- SaveStringKJ (save .txt caption files) ---
    ids["save_captions"] = wb.add_node(
        "SaveStringKJ", [x + 1320, y + 500], [400, 150],
        widgets_values=["img", "output/my_character_dataset", ".txt"],
        title="Save Captions (.txt)",
        properties={"cnr_id": "comfyui-kjnodes", "Node name for S&R": "SaveStringKJ"},
        mode=mode)
    wb.add_input(ids["save_captions"], "string", "STRING")
    wb.add_input(ids["save_captions"], "filename_prefix", "STRING")
    wb.add_input(ids["save_captions"], "output_folder", "STRING")
    wb.add_output(ids["save_captions"], "STRING", "STRING")

    wb.connect(ids["join_strings"], 0, ids["save_captions"], 0, "STRING")

    return ids


def main():
    wb = WorkflowBuilder()

    # =============================================
    # SHARED INFRASTRUCTURE
    # =============================================
    shared = build_shared_infrastructure(wb)

    wb.add_group("Shared Models & Parameters", [70, 60, 500, 3110], color="#444444")

    # =============================================
    # PART 1: GENERATION GROUPS
    # =============================================
    # Layout: groups arranged in columns, 2 rows
    # Each group occupies ~900px width, ~1200px height

    generation_groups = [
        # Group 1 — Turnaround Sheet
        {
            "num": 1,
            "name": "01_turnaround",
            "prompt": "Create a character turnaround sheet showing this person from front view, side view, back view, and three-quarter view. Full body. White background.",
            "image_source": "input_image",
        },
        # Group 2 — Portrait (clean)
        {
            "num": 2,
            "name": "02_portrait",
            "prompt": "Create a portrait photo of this person, front-facing, white background, neutral expression, professional headshot.",
            "image_source": "input_image",
            "extra_set_name": "portrait_output",  # Output used by emotion groups
        },
        # Group 3 — Close-up
        {
            "num": 3,
            "name": "03_closeup",
            "prompt": "Create an extreme close-up portrait of this person's face, showing detailed eyes and skin texture, studio lighting.",
            "image_source": "input_image",
        },
        # Group 4 — T-pose
        {
            "num": 4,
            "name": "04_tpose",
            "prompt": "Full body photo of this person in a T-pose, arms extended horizontally, white background.",
            "image_source": "input_image",
        },
        # Group 5 — Sitting
        {
            "num": 5,
            "name": "05_sitting",
            "prompt": "This person sitting casually on a chair, relaxed pose, indoor setting with soft lighting.",
            "image_source": "input_image",
        },
        # Group 6 — Standing side
        {
            "num": 6,
            "name": "06_standing_side",
            "prompt": "Side view of this person standing naturally with hands at sides, full body, white background.",
            "image_source": "input_image",
        },
        # Group 7 — Back view
        {
            "num": 7,
            "name": "07_back_view",
            "prompt": "Back view of this person standing naturally, full body, showing outfit from behind, white background.",
            "image_source": "input_image",
        },
        # Group 8 — Walking
        {
            "num": 8,
            "name": "08_walking",
            "prompt": "This person walking through a park, golden hour lighting, natural setting, full body mid-stride.",
            "image_source": "input_image",
        },
        # Group 9 — Happy
        {
            "num": 9,
            "name": "09_happy",
            "prompt": "Make this person look genuinely happy with a bright warm smile, joyful eyes.",
            "image_source": "portrait_output",
        },
        # Group 10 — Surprised
        {
            "num": 10,
            "name": "10_surprised",
            "prompt": "Make this person look surprised with wide eyes and slightly open mouth.",
            "image_source": "portrait_output",
        },
        # Group 11 — Angry
        {
            "num": 11,
            "name": "11_angry",
            "prompt": "Make this person look angry with furrowed brows, intense gaze, and clenched jaw.",
            "image_source": "portrait_output",
        },
        # Group 12 — Sad
        {
            "num": 12,
            "name": "12_sad",
            "prompt": "Make this person look sad with a melancholic expression, downcast eyes.",
            "image_source": "portrait_output",
        },
        # Group 13 — Laughing
        {
            "num": 13,
            "name": "13_laughing",
            "prompt": "Make this person laughing out loud with genuine amusement, open mouth laugh.",
            "image_source": "portrait_output",
        },
        # Group 14 — Contemplative
        {
            "num": 14,
            "name": "14_contemplative",
            "prompt": "Make this person look contemplative with a thoughtful gaze, slightly furrowed brow, looking into the distance.",
            "image_source": "portrait_output",
        },
        # Group 15 — Virtual Try-On 1
        {
            "num": 15,
            "name": "15_tryon_1",
            "prompt": "Make this person wear the outfit shown in the reference image. Keep the person's face and body identical.",
            "image_source": "input_image",
            "image2_source": "clothing_ref",
        },
        # Group 16 — Virtual Try-On 2
        {
            "num": 16,
            "name": "16_tryon_2",
            "prompt": "Make this person wear the clothing from the reference image. Maintain the person's identity perfectly.",
            "image_source": "input_image",
            "image2_source": "clothing_ref",
        },
        # Group 17 — Pose Transfer 1
        {
            "num": 17,
            "name": "17_pose_1",
            "prompt": "Make this person assume the exact pose shown in the reference image. Keep the person's appearance identical.",
            "image_source": "input_image",
            "image2_source": "pose_ref",
        },
        # Group 18 — Pose Transfer 2
        {
            "num": 18,
            "name": "18_pose_2",
            "prompt": "Make this person match the body position in the reference image. Preserve the person's identity and clothing.",
            "image_source": "input_image",
            "image2_source": "pose_ref",
        },
    ]

    # Layout constants — wider spacing so nodes don't overlap
    col_width = 1100
    row_height = 1400
    start_x = 700
    start_y = 100
    cols_per_row = 6

    group_ids = {}
    for g in generation_groups:
        idx = g["num"] - 1
        col = idx % cols_per_row
        row = idx // cols_per_row
        x = start_x + col * col_width
        y = start_y + row * row_height

        group_node_ids = build_generation_group(
            wb,
            group_num=g["num"],
            group_name=g["name"],
            prompt=g["prompt"],
            x_offset=x,
            image_source=g["image_source"],
            image2_source=g.get("image2_source"),
            save_prefix="my_character",
            extra_set_name=g.get("extra_set_name"),
        )
        group_ids[g["num"]] = group_node_ids

        # Add visual group
        group_label = f"Group {g['num']}: {g['name']}"
        # Color code by type
        if g["num"] <= 3:
            color = "#3b5998"  # blue - core views
        elif g["num"] <= 8:
            color = "#2d8659"  # green - poses
        elif g["num"] <= 14:
            color = "#8b4513"  # brown - emotions
        elif g["num"] <= 16:
            color = "#6a0dad"  # purple - try-on
        else:
            color = "#b8860b"  # gold - pose transfer
        wb.add_group(group_label, [x - 20, y - 30, col_width - 40, row_height - 60], color=color)

    # =============================================
    # PART 2: DATASET PREPARATION (muted by default)
    # =============================================
    part2_x = start_x
    part2_y = start_y + 3 * row_height + 200  # Below all Part 1 groups
    part2_ids = build_part2_dataset_prep(wb, part2_x, part2_y)

    wb.add_group("PART 2: Dataset Preparation (Unmute to run)",
                 [part2_x - 20, part2_y - 30, 1900, 750],
                 color="#cc3333")

    # Build and output
    workflow = wb.build()
    return workflow


if __name__ == "__main__":
    workflow = main()
    output_path = "character_dataset_creator.json"
    with open(output_path, "w") as f:
        json.dump(workflow, f, indent=2)
    print(f"Workflow saved to {output_path}")
    print(f"Total nodes: {len(workflow['nodes'])}")
    print(f"Total links: {len(workflow['links'])}")
    print(f"Total groups: {len(workflow['groups'])}")
