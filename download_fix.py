from roboflow import Roboflow
import os

api_key = "8GeCUXQU6JPzxMaYOe4m"
rf = Roboflow(api_key=api_key)
project = rf.workspace("mostafinafis").project("road-sign-detection-in-bd")
dataset = project.version(1).download("yolov8", location="./BRSSD_NEW")

print(f"Dataset downloaded to: {dataset.location}")
print("Listing contents:")
for root, dirs, files in os.walk(dataset.location):
    print(f"{root}: {len(files)} files")
