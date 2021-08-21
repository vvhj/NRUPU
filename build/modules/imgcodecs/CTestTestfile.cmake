# CMake generated Testfile for 
# Source directory: /root/work/Ultra-Fast-Lane-Detection-master/opencv-master/modules/imgcodecs
# Build directory: /root/work/Ultra-Fast-Lane-Detection-master/build/modules/imgcodecs
# 
# This file includes the relevant testing commands required for 
# testing this directory and lists subdirectories to be tested as well.
add_test(opencv_test_imgcodecs "/root/work/Ultra-Fast-Lane-Detection-master/build/bin/opencv_test_imgcodecs" "--gtest_output=xml:opencv_test_imgcodecs.xml")
set_tests_properties(opencv_test_imgcodecs PROPERTIES  LABELS "Main;opencv_imgcodecs;Accuracy" WORKING_DIRECTORY "/root/work/Ultra-Fast-Lane-Detection-master/build/test-reports/accuracy" _BACKTRACE_TRIPLES "/root/work/Ultra-Fast-Lane-Detection-master/opencv-master/cmake/OpenCVUtils.cmake;1707;add_test;/root/work/Ultra-Fast-Lane-Detection-master/opencv-master/cmake/OpenCVModule.cmake;1311;ocv_add_test_from_target;/root/work/Ultra-Fast-Lane-Detection-master/opencv-master/modules/imgcodecs/CMakeLists.txt;166;ocv_add_accuracy_tests;/root/work/Ultra-Fast-Lane-Detection-master/opencv-master/modules/imgcodecs/CMakeLists.txt;0;")
add_test(opencv_perf_imgcodecs "/root/work/Ultra-Fast-Lane-Detection-master/build/bin/opencv_perf_imgcodecs" "--gtest_output=xml:opencv_perf_imgcodecs.xml")
set_tests_properties(opencv_perf_imgcodecs PROPERTIES  LABELS "Main;opencv_imgcodecs;Performance" WORKING_DIRECTORY "/root/work/Ultra-Fast-Lane-Detection-master/build/test-reports/performance" _BACKTRACE_TRIPLES "/root/work/Ultra-Fast-Lane-Detection-master/opencv-master/cmake/OpenCVUtils.cmake;1707;add_test;/root/work/Ultra-Fast-Lane-Detection-master/opencv-master/cmake/OpenCVModule.cmake;1213;ocv_add_test_from_target;/root/work/Ultra-Fast-Lane-Detection-master/opencv-master/modules/imgcodecs/CMakeLists.txt;174;ocv_add_perf_tests;/root/work/Ultra-Fast-Lane-Detection-master/opencv-master/modules/imgcodecs/CMakeLists.txt;0;")
add_test(opencv_sanity_imgcodecs "/root/work/Ultra-Fast-Lane-Detection-master/build/bin/opencv_perf_imgcodecs" "--gtest_output=xml:opencv_perf_imgcodecs.xml" "--perf_min_samples=1" "--perf_force_samples=1" "--perf_verify_sanity")
set_tests_properties(opencv_sanity_imgcodecs PROPERTIES  LABELS "Main;opencv_imgcodecs;Sanity" WORKING_DIRECTORY "/root/work/Ultra-Fast-Lane-Detection-master/build/test-reports/sanity" _BACKTRACE_TRIPLES "/root/work/Ultra-Fast-Lane-Detection-master/opencv-master/cmake/OpenCVUtils.cmake;1707;add_test;/root/work/Ultra-Fast-Lane-Detection-master/opencv-master/cmake/OpenCVModule.cmake;1214;ocv_add_test_from_target;/root/work/Ultra-Fast-Lane-Detection-master/opencv-master/modules/imgcodecs/CMakeLists.txt;174;ocv_add_perf_tests;/root/work/Ultra-Fast-Lane-Detection-master/opencv-master/modules/imgcodecs/CMakeLists.txt;0;")