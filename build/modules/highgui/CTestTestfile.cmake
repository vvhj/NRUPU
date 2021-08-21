# CMake generated Testfile for 
# Source directory: /root/work/Ultra-Fast-Lane-Detection-master/opencv-master/modules/highgui
# Build directory: /root/work/Ultra-Fast-Lane-Detection-master/build/modules/highgui
# 
# This file includes the relevant testing commands required for 
# testing this directory and lists subdirectories to be tested as well.
add_test(opencv_test_highgui "/root/work/Ultra-Fast-Lane-Detection-master/build/bin/opencv_test_highgui" "--gtest_output=xml:opencv_test_highgui.xml")
set_tests_properties(opencv_test_highgui PROPERTIES  LABELS "Main;opencv_highgui;Accuracy" WORKING_DIRECTORY "/root/work/Ultra-Fast-Lane-Detection-master/build/test-reports/accuracy" _BACKTRACE_TRIPLES "/root/work/Ultra-Fast-Lane-Detection-master/opencv-master/cmake/OpenCVUtils.cmake;1707;add_test;/root/work/Ultra-Fast-Lane-Detection-master/opencv-master/cmake/OpenCVModule.cmake;1311;ocv_add_test_from_target;/root/work/Ultra-Fast-Lane-Detection-master/opencv-master/modules/highgui/CMakeLists.txt;165;ocv_add_accuracy_tests;/root/work/Ultra-Fast-Lane-Detection-master/opencv-master/modules/highgui/CMakeLists.txt;0;")
