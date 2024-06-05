# TresPass_StoreGuard

TresPassGuard is a real-time security system designed for retail environments, particularly gold stores. It uses advanced object detection, line-crossing detection, 
and alert mechanisms to notify store owners of unauthorized movements. TresPassGuard supports multiple RTSP cameras for comprehensive surveillance.

## Features

- **Object Detection**: Utilizes YOLOv8 for accurate person detection.
- **Line-Crossing Detection**: Detects when a person crosses a predefined boundary using Shapely for intersection detection.
- **Real-Time Alerts**: Triggers an alarm to alert the shop owner of a potential trespasser.
- **Visualization**: Uses OpenCV to display detected objects and crossing events on the camera feed.

## Components

- **YOLOv8**: For object detection.
- **Shapely**: For geometric operations to detect line crossings.
- **Supervision**: To manage and process detection events.
- **OpenCV**: For image processing and visualization.

## Requirements

- ultralytics
- opencv-python==4.10.0.82
- supervision==0.3.0
- shapely==2.0.1
- chardet
- pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
  
## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/deepaks11/TresPass_StoreGuard
    cd TresPass_StoreGuard
    ```


2. Set Up Conda Environment
   conda create --name (env name) python=3.10
   conda activate intent-identification

3. Install Dependencies
   pip install -r requirements.txt

4. Run the Project
   python rtsp_stream.py

## Usage

1. **Configure the detection zone**: Edit the script to set up the boundary line coordinates according to your store layout.

2. **Run the script**:
    ```bash
    python rtsp_stream.py
    ```

3. The system will start detecting people and trigger an alert if someone crosses the predefined line.

## Example

Here's an example of how the system detects and alerts on trespassing:

![TresPassGuard Example](./path_to_your_image.png)

## Contributing

We welcome contributions to enhance TresPassGuard. If you have any ideas or improvements, please open an issue or submit a pull request.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.

## Acknowledgements

- Thanks to the developers of YOLOv8, Shapely, Supervision, and OpenCV for their excellent tools.
