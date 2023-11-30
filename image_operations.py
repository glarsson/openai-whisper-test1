def webcam_capture():
    cap = cv2.VideoCapture(cam_device_index)
    
    # Check if the camera is opened successfully
    if not cap.isOpened():
        print("Unable to open camera")
        return
    
    while True:
        ret, frame = cap.read()
        
        # Check if the frame was read successfully
        if not ret:
            break

        # Convert the OpenCV BGR image to RGB (PIL format)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Resize the image to 800x600
        resized_frame = cv2.resize(rgb_frame, (1280, 720))

        # Acquire the lock before writing to the file
        with lock:
            # Write the image to the output file
            output_filename = 'temp/vision.png'
            ret, buffer = cv2.imencode(".png", cv2.cvtColor(resized_frame, cv2.COLOR_RGB2BGR))
            if ret:
                with open(output_filename, 'wb') as out_file:  # Define the out_file variable
                    out_file.write(buffer)
                    out_file.close()  # Close the file after writing
                    if not os.path.exists('temp/vision2.png'):
                        shutil.copy(output_filename, 'temp/vision2.png')
                    time.sleep(0.2)

            # Only copy the file if it does not exist - this is to prevent the processing thread from reading an empty file
            # this whole thing is a really poor mans solution to threading and please don't judge me, it's only a PoC.


    # Release the camera when the loop is exited
    cap.release()