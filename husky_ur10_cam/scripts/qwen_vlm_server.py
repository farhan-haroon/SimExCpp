#!/usr/bin/env python3
"""Shared Qwen2.5-VL-7B-Instruct inference server - custom_interfaces/srv/
VlmQuery - so only one copy of the model (~16.6GB in bf16) is ever
resident, instead of every VLM caller loading its own.

Clients: ee_camera_environment_scan.py's vlm_guide() (phase 2 move
guidance) and active_perception_node.py (periodic chassis-camera scene
check), both via EnvironmentScanner/simple service-client calls to
"vlm_query". Neither builds prompts or parses responses here - this node
is a pure model-inference passthrough (images + prompt in, raw text out);
each caller owns its own prompt wording and response parsing, same as
before the refactor.

The service callback runs in the default (mutually exclusive) callback
group, so concurrent requests from multiple clients serialize naturally -
the GPU only ever runs one generate() call at a time.

Usage:
    ros2 run husky_ur10_cam qwen_vlm_server.py
"""
import os
import sys
import time

import rclpy
from rclpy.node import Node
from rclpy.executors import ExternalShutdownException

# ee_camera_environment_scan.py is a sibling script, not an installed
# package - add its directory to sys.path so it can be imported directly
# (same trick object_search_action_server.py uses).
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ee_camera_environment_scan import (  # noqa: E402
    DEFAULT_QWEN_MODEL,
    _image_msg_to_pil,
    install_shutdown_handler,
)

from custom_interfaces.srv import VlmQuery  # noqa: E402

_qwen_model = None
_qwen_processor = None


def _get_qwen(model_id):
    global _qwen_model, _qwen_processor
    if _qwen_model is None:
        import torch
        from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
        # device_map="auto" (not a single fixed cuda:N) - the 7B checkpoint
        # is ~16.6GB in bf16, which doesn't reliably fit on one 16GB card
        # alongside whatever else is resident (Gazebo/rviz, etc). "auto"
        # lets accelerate shard it across both GPUs as needed.
        _qwen_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_id, torch_dtype=torch.bfloat16, device_map="auto",
            local_files_only=True,
        )
        _qwen_processor = AutoProcessor.from_pretrained(model_id, local_files_only=True)
    return _qwen_model, _qwen_processor


class QwenVlmServer(Node):
    def __init__(self):
        super().__init__("qwen_vlm_server")
        self.declare_parameter("qwen_model", DEFAULT_QWEN_MODEL)
        self._service = self.create_service(VlmQuery, "vlm_query", self._handle_query)

    def _handle_query(self, request, response):
        import torch

        self.get_logger().info(
            f"vlm_query: {len(request.images)} image(s), prompt={request.prompt[:80]!r}..."
        )
        t0 = time.time()

        model_id = self.get_parameter("qwen_model").value
        was_loaded = _qwen_model is not None
        model, processor = _get_qwen(model_id)
        if not was_loaded:
            self.get_logger().info(f"Model loaded in {time.time() - t0:.1f}s")

        images = [_image_msg_to_pil(img) for img in request.images]
        content = [{"type": "image", "image": img} for img in images]
        content.append({"type": "text", "text": request.prompt})
        messages = [{"role": "user", "content": content}]

        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = processor(text=[text], images=images, return_tensors="pt").to(model.device)

        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=request.max_new_tokens, do_sample=False)
        gen = out[:, inputs["input_ids"].shape[1]:]
        response.response = processor.batch_decode(gen, skip_special_tokens=True)[0]
        self.get_logger().info(
            f"vlm_query: response={response.response!r} ({time.time() - t0:.1f}s total)"
        )
        return response


def _default_params_file():
    """husky_ur10_cam/config/all_params.yaml, if installed - see its
    header comment. Auto-loaded below so `ros2 run` alone picks it up."""
    try:
        from ament_index_python.packages import get_package_share_directory
        path = os.path.join(
            get_package_share_directory("husky_ur10_cam"), "config", "all_params.yaml")
        return path if os.path.isfile(path) else None
    except Exception:
        return None


def main(args=None):
    install_shutdown_handler()
    argv = list(sys.argv if args is None else args)
    if "--params-file" not in argv:
        default_params = _default_params_file()
        if default_params:
            argv += ["--ros-args", "--params-file", default_params]

    rclpy.init(args=argv)
    node = QwenVlmServer()
    node.get_logger().info(
        f"vlm_query service ready (model: {node.get_parameter('qwen_model').value}) "
        "- first request will load the model, which can take a while"
    )
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
