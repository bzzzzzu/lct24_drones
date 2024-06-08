import cv2

def test_camera():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Ошибка: Не удалось открыть камеру.")
        return

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Ошибка: Не удалось захватить кадр.")
                break

            cv2.imshow('Test Camera', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    test_camera()