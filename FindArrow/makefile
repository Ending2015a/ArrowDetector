CC = g++
CFLAGS = -g -Wall -I/usr/include -I/home/ubuntu/librealsense-armhf/examples/include -L/usr/local/lib/
opencvlibs = -lopencv_core -lopencv_imgproc -lopencv_highgui -lopencv_ml -lopencv_video -lopencv_features2d -lopencv_calib3d -lopencv_objdetect -lopencv_contrib -lopencv_legacy -lopencv_stitching

Arrow: Source.o
	${CC} ${CFLAGS} Source.o -o $@ ${opencvlibs} -lrealsense
Source.o: Source.cpp
	${CC} ${CFLAGS} -c $< -o $@ ${opencvlibs} -lrealsense
clean:
	@rm -rf *.o