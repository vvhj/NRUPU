# CMake generated Testfile for 
# Source directory: /root/work/Ultra-Fast-Lane-Detection-master/opencv-master/modules/flann
# Build directory: /root/work/Ultra-Fast-Lane-Detection-master/build/modules/flann
# 
# This file includes the relevant testing commands required for 
# testing this directory and lists subdirectories to be tested as well.
add_test(opencv_test_flann "/root/work/Ultra-Fast-Lane-Detection-master/build/bin/opencv_test_flann" "--gtest_output=xml:opencv_test_flann.xml")
set_tests_properties(opencv_test_flann PROPERTIES  LABELS "Main;opencv_flann;Accuracy" WORKING_DIRECTORY "/root/work/Ultra-Fast-Lane-Detection-master/build/test-reports/accuracy" _BACKTRACE_TRIPLES "/root/work/Ultra-Fast-Lane-Detection-master/opencv-master/cmake/OpenCVUtils.cmake;1707;add_test;/root/work/Ultra-Fast-Lane-Detection-master/opencv-master/cmake/OpenCVModule.cmake;1311;ocv_add_test_from_target;/root/work/Ultra-Fast-Lane-Detection-master/opencv-master/cmake/OpenCVModule.cmake;1075;ocv_add_accuracy_tests;/root/work/Ultra-Fast-Lane-Detection-master/opencv-master/modules/flann/CMakeLists.txt;2;ocv_define_module;/root/work/Ultra-Fast-Lane-Detection-master/opencv-master/modules/flann/CMakeLists.txt;0;")
