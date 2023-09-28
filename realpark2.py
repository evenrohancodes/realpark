# Import necessary libraries
from flask import Flask, render_template, Response
import yaml
import numpy as np
import cv2

app = Flask(__name__)

# Define your configuration and OpenCV capture here
area = 12
yaml_file = r"realpark/datasets/parking2.yml"
fn_out = r"../datasets/output.avi"
config = {
    'save_video': False,
    'text_overlay': True,
    'parking_overlay': True,
    'parking_id_overlay': False,
    'parking_detection': True,
    'min_area_motion_contour': 60,
    'park_sec_to_wait': 1,
    'start_frame': 0,
}

# Set capture device 0 for the built-in camera, 1 for a USB camera
cap = cv2.VideoCapture(0)

# Set the path to your MP4 file
# video_file = 'realpark\sample.mp4'
# cap = cv2.VideoCapture(video_file)

if config['save_video']:
    fourcc = cv2.VideoWriter_fourcc('D','I','V','X')  # You can adjust the codec here
    out = cv2.VideoWriter(fn_out, fourcc, 25.0, (640, 480))  # Adjust frame size as needed

def load_parking_data(yaml_file):
    with open(yaml_file, 'r') as stream:
        parking_data = yaml.safe_load(stream)
    parking_contours = []
    parking_bounding_rects = []
    parking_mask = []
    for park in parking_data:
        points = np.array(park['points'])
        rect = cv2.boundingRect(points)
        points_shifted = points.copy()
        points_shifted[:, 0] = points[:, 0] - rect[0]
        points_shifted[:, 1] = points[:, 1] - rect[1]
        parking_contours.append(points)
        parking_bounding_rects.append(rect)
        mask = cv2.drawContours(np.zeros((rect[3], rect[2]), dtype=np.uint8), [points_shifted], contourIdx=-1, color=255, thickness=-1, lineType=cv2.LINE_8)
        mask = mask == 255
        parking_mask.append(mask)
    return parking_data, parking_contours, parking_bounding_rects, parking_mask

parking_data, parking_contours, parking_bounding_rects, parking_mask = load_parking_data(yaml_file)

# Initialize global variables for parking status
parking_status = [False] * len(parking_data)
parking_buffer = [None] * len(parking_data)

# Define a route to display the video stream
@app.route('/')
def index():
    return render_template('index.html')

# Define a route for streaming video
#section1:
def gen():
    global area
    print ("Program Successfull to detect the presence of cars in the parking area, wait for 5 seconds before the status changes.Frame size is large, 960x720")
    while True:
        spot=0
        occupied=0
        video_cur_pos = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
        success, frame = cap.read()
        if not success:
            print("Capture error")
            break

        # Process the frame as per your existing code
        frame_blur = cv2.GaussianBlur(frame.copy(), (5, 5), 3)
        frame_gray = cv2.cvtColor(frame_blur, cv2.COLOR_BGR2GRAY)
        frame=frame.copy()

        # Add your parking detection logic here (similar to your existing code)
        if config['parking_detection']:
            for ind, park in enumerate(parking_data):
                points = np.array(park['points'])
                rect = parking_bounding_rects[ind]
                roi_gray = frame_gray[rect[1]:(rect[1] + rect[3]), rect[0]:(rect[0] + rect[2])]  # crop roi for faster calculation

                # Calculate standard deviation and mean of the ROI
                std_dev = np.std(roi_gray)
                mean_val = np.mean(roi_gray)

                # Determine parking status based on the calculated values
                status = std_dev < 22 and mean_val > 53

                # If detected a change in parking status, save the current time
                if status != parking_status[ind] and parking_buffer[ind] is None:
                    parking_buffer[ind] = video_cur_pos

                # If status is still different than the one saved and counter is open
                elif status != parking_status[ind] and parking_buffer[ind] is not None:
                    if video_cur_pos - parking_buffer[ind] > config['park_sec_to_wait']:
                        print(ind + 1)
                        print(status)
                        area = ind + 1
                        parking_status[ind] = status
                        parking_buffer[ind] = None

                # If status is still same and counter is open
                elif status == parking_status[ind] and parking_buffer[ind] is not None:
                    parking_buffer[ind] = None

        # Draw Overlay (similar to your existing code)
        if config['parking_overlay']:
            for ind, park in enumerate(parking_data):
                points = np.array(park['points'])
                if parking_status[ind]:
                    color = (0,255,0)
                    spot = spot + 1
                else:
                    color = (0,0,255)
                    occupied = occupied + 1
                cv2.drawContours(frame, [points], contourIdx=-1, color=color, thickness=2, lineType=cv2.LINE_8)
                moments = cv2.moments(points)
                centroid = (int(moments['m10'] / moments['m00']) - 3, int(moments['m01'] / moments['m00']) + 3)
                cv2.putText(frame, str(park['id']), (centroid[0] + 1, centroid[1] + 1), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                            (255, 255, 255), 1, cv2.LINE_AA)
                cv2.putText(frame, str(park['id']), (centroid[0] - 1, centroid[1] - 1), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                            (255, 255, 255), 1, cv2.LINE_AA)
                cv2.putText(frame, str(park['id']), (centroid[0] + 1, centroid[1] - 1), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                            (255, 255, 255), 1, cv2.LINE_AA)
                cv2.putText(frame, str(park['id']), (centroid[0] - 1, centroid[1] + 1), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                            (255, 255, 255), 1, cv2.LINE_AA)
                cv2.putText(frame, str(park['id']), centroid, cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1,
                            cv2.LINE_AA)
                
        if config['text_overlay']:
                    #cv2.rectangle(frame_out, (1, 5), (280, 70),(255,255,255), 85) 
                cv2.rectangle(frame, (1, 5), (300, 70),(0,255,0), 2)
                str_on_frame = "Parking Area Status:"
                cv2.putText(frame, str_on_frame, (5,20), cv2.FONT_HERSHEY_SIMPLEX ,0.5, (0,0,255), 2, cv2.LINE_AA)
                str_on_frame = "Empty space = %d, Occupied = %d" % (spot, occupied)
                cv2.putText(frame, str_on_frame, (5,40), cv2.FONT_HERSHEY_SIMPLEX ,0.5, (0,0,255), 2, cv2.LINE_AA)
                str_on_frame = "Last change area: " + str(area)
                cv2.putText(frame, str_on_frame, (5,60), cv2.FONT_HERSHEY_SIMPLEX ,0.5, (255,255,0), 1, cv2.LINE_AA)
        
        # Convert the frame to JPEG format for streaming
        success, jpeg = cv2.imencode('.jpg', frame)
        frame = jpeg.tobytes()

        # Yield the frame as part of the video stream
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()
    if config['save_video']: out.release()
    cv2.destroyAllWindows()  

@app.route('/video_feed')
def video_feed():
    return Response(gen(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)
