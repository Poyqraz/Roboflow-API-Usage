from roboflow import Roboflow
from inference_sdk import InferenceHTTPClient
from inference import InferencePipeline
from inference.core.interfaces.stream.sinks import render_boxes


# Pulling model and project information from Roboflow
rf = Roboflow(api_key="XXXXXXXXXX")  #Write your roboflow api key where it is written with XX.
project = rf.workspace().project("YYYYYY")  #write the name of your project where it is written with YY.
model = project.version("W").model  #write the project version. Usually writes as 1 if you create a project for the first time


# Starting InferencePipeline using InferenceHTTPClient
CLIENT = InferenceHTTPClient(api_url="api url", api_key="XXXXXXXXXX")
# Take roboflow workspace url is as api url


# Starting InferencePipeline
pipeline = InferencePipeline.init(
    model_id=("QQQQQQQQ"),  # Name of the trained model to be used
    video_reference=0,  # Specify the video path or device ID (usually 0 or 1 for built-in webcams).
    on_prediction=render_boxes,  # Function to run after each prediction.
#    device="gpu"   # Turns on GPU utilization (includes Nvidia video cards)
)

# Starting Pipeline
pipeline.start()
pipeline.join()

# for terminal using === python real_time_detect.py