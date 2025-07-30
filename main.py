import cv2
import numpy as np

# Configuration
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
ABANDON_THRESHOLD_FRAMES = 100
MIN_CONTOUR_AREA = 500

background_model = None
object_history = {}
frame_counter = 0

def get_center(x, y, w, h):
    return (x + w // 2, y + h // 2)

def draw_box(frame, x, y, w, h, label, color):
    frame[y:y+2, x:x+w] = color
    frame[y+h-2:y+h, x:x+w] = color
    frame[y:y+h, x:x+2] = color
    frame[y:y+h, x+w-2:x+w] = color

def to_grayscale(frame):
    return (0.299 * frame[:, :, 2] + 0.587 * frame[:, :, 1] + 0.114 * frame[:, :, 0]).astype(np.uint8)

def fast_gaussian_blur(image, kernel_size=5, sigma=1.0):
    k = kernel_size // 2
    ax = np.arange(-k, k+1)
    gaussian = np.exp(-0.5 * (ax / sigma) ** 2)
    kernel = gaussian / gaussian.sum()
    temp = np.apply_along_axis(lambda m: np.convolve(m, kernel, mode='same'), axis=1, arr=image)
    blurred = np.apply_along_axis(lambda m: np.convolve(m, kernel, mode='same'), axis=0, arr=temp)
    return blurred.astype(np.uint8)

def threshold(image, thresh_value):
    return np.where(image > thresh_value, 255, 0).astype(np.uint8)

def morphological_opening(mask):
    eroded = cv2.erode(mask, np.ones((3,3), np.uint8), iterations=1)
    dilated = cv2.dilate(eroded, np.ones((3,3), np.uint8), iterations=1)
    return dilated

def find_contours(binary_image):
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    result = []
    for cnt in contours:
        if cv2.contourArea(cnt) >= MIN_CONTOUR_AREA:
            x, y, w, h = cv2.boundingRect(cnt)
            result.append((x, y, w, h))
    return result

def process_video(video_path):
    global background_model, frame_counter
    cap = cv2.VideoCapture(video_path)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))
        frame_counter += 1

        gray = to_grayscale(frame)
        blur = fast_gaussian_blur(gray)

        if background_model is None:
            background_model = blur.copy()
            continue

        diff = cv2.absdiff(background_model, blur)
        thresh = threshold(diff, 30)
        mask = morphological_opening(thresh)
        contours = find_contours(mask)

        current_objects = {}
        for x, y, w, h in contours:
            center = get_center(x, y, w, h)
            current_objects[center] = (x, y, w, h)

            if center in object_history:
                object_history[center]['static_count'] += 1
            else:
                object_history[center] = {'static_count': 1, 'last_position': center}

            label = "Abandoned" if object_history[center]['static_count'] >= ABANDON_THRESHOLD_FRAMES else "Tracking"
            color = (0, 0, 255) if label == "Abandoned" else (255, 0, 0)
            draw_box(frame, x, y, w, h, label, color)

        to_remove = []
        for center in object_history:
            if center not in current_objects:
                object_history[center]['static_count'] -= 1
                if object_history[center]['static_count'] <= 0:
                    to_remove.append(center)
        for key in to_remove:
            del object_history[key]

        cv2.imshow("Abandoned Object Detection", frame)
        cv2.imshow("Foreground Mask", mask)

        if cv2.waitKey(30) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

# Run the detection
process_video("data/video11.mp4")
