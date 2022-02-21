import cv2

print("Before URL")
try :
    cap = cv2.VideoCapture('rtsp://admin:SFYZEV@78.113.98.174:554/H.264')
except: print('error')
print("After URL")

if __name__=='__main__':
    while True:

        print('About to start the Read command')
        ret, frame = cap.read()
        print('About to show frame of Video.')
        cv2.startWindowThread()
        cv2.namedWindow("preview")
        cv2.imshow("prewiew",frame)
        cv2.waitKey(1)

        print('Running..')

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()