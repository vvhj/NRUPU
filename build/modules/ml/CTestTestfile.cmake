# CMake generated Testfile for 
# Source directory: /root/work/Ultra-Fast-Lane-Detection-master/opencv-master/modules/ml
# Build directory: /root/work/Ultra-Fast-Lane-Detection-master/build/modules/ml
# 
# This file includes the relevant testing commands required for 
# testing this directory and lists subdirectories to be tested as well.
add_test(opencv_test_ml "/root/work/Ultra-Fast-Lane-Detection-master/build/bin/opencv_test_ml" "--gtest_output=xml:opencv_test_ml.xml")
set_tests_properties(opencv_test_ml PROPERTIES  LABELS "Main;opencv_ml;Accuracy" WORKING_DIRECTORY "/root/work/Ultra-Fast-Lane-Detection-master/build/test-reports/accuracy" _BACKTRACE_TRIPLES "/root/work/Ultra-Fast-Lane-Detection-master/opencv-master/cmake/OpenCVUtils.cmake;1707;add_test;/root/work/Ultra-Fast-Lane-Detection-master/opencv-master/cmake/OpenCVModule.cmake;1311;ocv_add_test_from_target;/root/work/Ultra-Fast-Lane-Detection-master/opencv-master/cmake/OpenCVModule.cmake;1075;ocv_add_accuracy_tests;/root/work/Ultra-Fast-Lane-Detection-master/opencv-master/modules/ml/CMakeLists.txt;2;ocv_define_module;/root/work/Ultra-Fast-Lane-Detection-master/opencv-master/modules/ml/CMakeLists.txt;0;")
