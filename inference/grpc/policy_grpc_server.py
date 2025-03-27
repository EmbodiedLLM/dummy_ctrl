import grpc
import torch
import numpy as np
import time
import os
import logging
from concurrent import futures
import sys
from typing import Dict, Any
import cv2

from proto import policy_pb2
from proto import policy_pb2_grpc

# Import the ACT policy model
from lerobot.common.policies.act.modeling_act import ACTPolicy

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load the ACT policy model
# Update this path to your model location
import argparse

# Set up argument parser
parser = argparse.ArgumentParser(description="Policy gRPC Server")
parser.add_argument("--model_path", type=str, required=True,help="Path to the pretrained policy model")
args = parser.parse_args()
PRETRAINED_POLICY_PATH = args.model_path

# Determine the device based on platform and availability
if torch.cuda.is_available():
    device = "cuda"  
elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    device = "mps"  # Use Metal Performance Shaders for Mac with Apple Silicon
else:
    device = "cpu"  # Fallback to CPU

logger.info(f"Using device: {device}")

class PolicyServicer(policy_pb2_grpc.PolicyServiceServicer):
    def __init__(self):
        super().__init__()
        self.policy = None
        self.load_policy()
        
    def load_policy(self):
        """Load the policy model from the pretrained path"""
        try:
            logger.info(f"Loading ACT policy from {PRETRAINED_POLICY_PATH}")
            self.policy = ACTPolicy.from_pretrained(PRETRAINED_POLICY_PATH)
            self.policy.to(device)
            self.policy.eval()  # Set to evaluation mode
            self.policy.reset()  # Reset policy state
            logger.info(f"Successfully loaded ACT policy")
        except Exception as e:
            self.policy = None
            # logger.error(f"Could not load ACT policy model: {e}")
            # logger.warning("Will use placeholder prediction (ones tensor)")
            raise e
    
    def reshape_image(self, flat_image, channels, height, width):
        """Reshape flat image data to tensor format [C, H, W]"""
        return torch.tensor(flat_image, dtype=torch.float).reshape(channels, height, width)
    
    def decode_image(self, encoded_image, image_format, height, width):
        """Decode compressed image data to tensor format [C, H, W]"""
        # Convert bytes to numpy array
        nparr = np.frombuffer(encoded_image, np.uint8)
        
        # Decode image
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError(f"Failed to decode image with format {image_format}")
        
        # Convert BGR to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Normalize to [0, 1]
        img = img.astype(np.float32) / 255.0
        
        # Convert to CHW format (channels, height, width)
        img = img.transpose(2, 0, 1)
        
        # Convert to tensor
        return torch.tensor(img, dtype=torch.float)
    
    def Predict(self, request, context):
        """Handle prediction requests"""
        try:
            # Check if we have encoded image or raw image data
            if request.encoded_image:
                logger.info(f"Received encoded image in {request.image_format} format, size: {len(request.encoded_image)/1024:.2f}KB")
                # Decode the encoded image
                image_tensor = self.decode_image(
                    request.encoded_image,
                    request.image_format,
                    request.image_height,
                    request.image_width
                )
            else:
                # Use the original flat image data
                logger.info("Received flat image data")
                image_tensor = self.reshape_image(
                    request.image, 
                    request.image_channels, 
                    request.image_height, 
                    request.image_width
                )
            
            # Convert state data to tensor
            state_tensor = torch.tensor([request.state], dtype=torch.float)
            
            logger.info(f"Processed input - Image shape: {image_tensor.shape}, State shape: {state_tensor.shape}")
            
            # Validate input shapes
            if image_tensor.shape[0] != 3:
                context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
                context.set_details("Image must have 3 channels (RGB)")
                return policy_pb2.PredictResponse()
            
            if state_tensor.shape != (1, 7):
                context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
                context.set_details(f"State must have shape (1, 7), got {state_tensor.shape}")
                return policy_pb2.PredictResponse()
            
            # Move tensors to device
            image_tensor = image_tensor.to(device)
            state_tensor = state_tensor.to(device)
            
            # Create the policy input dictionary
            observation = {
                "observation.state": state_tensor,
                "observation.images.cam_wrist": image_tensor.unsqueeze(0),  # Add batch dimension
            }
            
            
            if self.policy is not None:
                # Perform the prediction
                start_time = time.perf_counter()
                with torch.inference_mode():
                    action = self.policy.select_action(observation)
                end_time = time.perf_counter()
                inference_time_ms = (end_time - start_time) * 1000  # Convert to milliseconds
                logger.info(f"Inference time: {inference_time_ms:.2f}ms")

                prediction = action.squeeze(0).to("cpu")
                logger.info(f"Prediction shape: {prediction.shape}")
            else:
                # Fallback to placeholder prediction
                prediction = torch.ones(7)
                
            
            # Create and return response
            response = policy_pb2.PredictResponse(
                prediction=prediction.tolist(),
                inference_time_ms=inference_time_ms
            )
            return response
            
        except Exception as e:
            logger.error(f"Error during prediction: {e}")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            return policy_pb2.PredictResponse()
    
    def HealthCheck(self, request, context):
        """Handle health check requests"""
        return policy_pb2.HealthCheckResponse(status="healthy")
    
    def GetModelInfo(self, request, context):
        """Handle model info requests"""
        if self.policy is not None:
            return policy_pb2.ModelInfoResponse(
                status="loaded",
                model_path=PRETRAINED_POLICY_PATH,
                device=device,
                input_features=str(self.policy.config.input_features) if hasattr(self.policy.config, "input_features") else "unknown",
                output_features=str(self.policy.config.output_features) if hasattr(self.policy.config, "output_features") else "unknown"
            )
        else:
            return policy_pb2.ModelInfoResponse(
                status="not_loaded",
                model_path="",
                device=device,
                message="Using placeholder prediction (ones tensor)"
            )


def serve():
    """Start the gRPC server"""
    # Create a gRPC server with 10 workers
    server = grpc.server(
        futures.ThreadPoolExecutor(max_workers=10),
        options=[
            ('grpc.max_send_message_length', 50 * 1024 * 1024),  # 50MB
            ('grpc.max_receive_message_length', 50 * 1024 * 1024)  # 50MB
        ]
    )
    
    # Add the PolicyServicer to the server
    policy_pb2_grpc.add_PolicyServiceServicer_to_server(
        PolicyServicer(), server
    )
    
    # Listen on port 50051
    server.add_insecure_port('[::]:50051')
    server.start()
    
    logger.info("Policy gRPC server started on port 50051")
    
    try:
        # Keep the server running until interrupted
        server.wait_for_termination()
    except KeyboardInterrupt:
        logger.info("Server shutting down...")
        server.stop(0)


if __name__ == "__main__":
    serve()
