# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.31

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Disable VCS-based implicit rules.
% : %,v

# Disable VCS-based implicit rules.
% : RCS/%

# Disable VCS-based implicit rules.
% : RCS/%,v

# Disable VCS-based implicit rules.
% : SCCS/s.%

# Disable VCS-based implicit rules.
% : s.%

.SUFFIXES: .hpux_make_needs_suffix_list

# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /opt/homebrew/bin/cmake

# The command to remove a file.
RM = /opt/homebrew/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /Users/aditshah/Desktop/PRCV/Project4/Task7

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /Users/aditshah/Desktop/PRCV/Project4/Task7/build

# Include any dependencies generated for this target.
include CMakeFiles/task7.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include CMakeFiles/task7.dir/compiler_depend.make

# Include the progress variables for this target.
include CMakeFiles/task7.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/task7.dir/flags.make

CMakeFiles/task7.dir/codegen:
.PHONY : CMakeFiles/task7.dir/codegen

CMakeFiles/task7.dir/task7.cpp.o: CMakeFiles/task7.dir/flags.make
CMakeFiles/task7.dir/task7.cpp.o: /Users/aditshah/Desktop/PRCV/Project4/Task7/task7.cpp
CMakeFiles/task7.dir/task7.cpp.o: CMakeFiles/task7.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/Users/aditshah/Desktop/PRCV/Project4/Task7/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/task7.dir/task7.cpp.o"
	/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/task7.dir/task7.cpp.o -MF CMakeFiles/task7.dir/task7.cpp.o.d -o CMakeFiles/task7.dir/task7.cpp.o -c /Users/aditshah/Desktop/PRCV/Project4/Task7/task7.cpp

CMakeFiles/task7.dir/task7.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/task7.dir/task7.cpp.i"
	/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /Users/aditshah/Desktop/PRCV/Project4/Task7/task7.cpp > CMakeFiles/task7.dir/task7.cpp.i

CMakeFiles/task7.dir/task7.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/task7.dir/task7.cpp.s"
	/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /Users/aditshah/Desktop/PRCV/Project4/Task7/task7.cpp -o CMakeFiles/task7.dir/task7.cpp.s

# Object files for target task7
task7_OBJECTS = \
"CMakeFiles/task7.dir/task7.cpp.o"

# External object files for target task7
task7_EXTERNAL_OBJECTS =

task7: CMakeFiles/task7.dir/task7.cpp.o
task7: CMakeFiles/task7.dir/build.make
task7: /opt/homebrew/lib/libopencv_gapi.4.11.0.dylib
task7: /opt/homebrew/lib/libopencv_stitching.4.11.0.dylib
task7: /opt/homebrew/lib/libopencv_alphamat.4.11.0.dylib
task7: /opt/homebrew/lib/libopencv_aruco.4.11.0.dylib
task7: /opt/homebrew/lib/libopencv_bgsegm.4.11.0.dylib
task7: /opt/homebrew/lib/libopencv_bioinspired.4.11.0.dylib
task7: /opt/homebrew/lib/libopencv_ccalib.4.11.0.dylib
task7: /opt/homebrew/lib/libopencv_dnn_objdetect.4.11.0.dylib
task7: /opt/homebrew/lib/libopencv_dnn_superres.4.11.0.dylib
task7: /opt/homebrew/lib/libopencv_dpm.4.11.0.dylib
task7: /opt/homebrew/lib/libopencv_face.4.11.0.dylib
task7: /opt/homebrew/lib/libopencv_freetype.4.11.0.dylib
task7: /opt/homebrew/lib/libopencv_fuzzy.4.11.0.dylib
task7: /opt/homebrew/lib/libopencv_hfs.4.11.0.dylib
task7: /opt/homebrew/lib/libopencv_img_hash.4.11.0.dylib
task7: /opt/homebrew/lib/libopencv_intensity_transform.4.11.0.dylib
task7: /opt/homebrew/lib/libopencv_line_descriptor.4.11.0.dylib
task7: /opt/homebrew/lib/libopencv_mcc.4.11.0.dylib
task7: /opt/homebrew/lib/libopencv_quality.4.11.0.dylib
task7: /opt/homebrew/lib/libopencv_rapid.4.11.0.dylib
task7: /opt/homebrew/lib/libopencv_reg.4.11.0.dylib
task7: /opt/homebrew/lib/libopencv_rgbd.4.11.0.dylib
task7: /opt/homebrew/lib/libopencv_saliency.4.11.0.dylib
task7: /opt/homebrew/lib/libopencv_sfm.4.11.0.dylib
task7: /opt/homebrew/lib/libopencv_signal.4.11.0.dylib
task7: /opt/homebrew/lib/libopencv_stereo.4.11.0.dylib
task7: /opt/homebrew/lib/libopencv_structured_light.4.11.0.dylib
task7: /opt/homebrew/lib/libopencv_superres.4.11.0.dylib
task7: /opt/homebrew/lib/libopencv_surface_matching.4.11.0.dylib
task7: /opt/homebrew/lib/libopencv_tracking.4.11.0.dylib
task7: /opt/homebrew/lib/libopencv_videostab.4.11.0.dylib
task7: /opt/homebrew/lib/libopencv_viz.4.11.0.dylib
task7: /opt/homebrew/lib/libopencv_wechat_qrcode.4.11.0.dylib
task7: /opt/homebrew/lib/libopencv_xfeatures2d.4.11.0.dylib
task7: /opt/homebrew/lib/libopencv_xobjdetect.4.11.0.dylib
task7: /opt/homebrew/lib/libopencv_xphoto.4.11.0.dylib
task7: /opt/homebrew/lib/libopencv_shape.4.11.0.dylib
task7: /opt/homebrew/lib/libopencv_highgui.4.11.0.dylib
task7: /opt/homebrew/lib/libopencv_datasets.4.11.0.dylib
task7: /opt/homebrew/lib/libopencv_plot.4.11.0.dylib
task7: /opt/homebrew/lib/libopencv_text.4.11.0.dylib
task7: /opt/homebrew/lib/libopencv_ml.4.11.0.dylib
task7: /opt/homebrew/lib/libopencv_phase_unwrapping.4.11.0.dylib
task7: /opt/homebrew/lib/libopencv_optflow.4.11.0.dylib
task7: /opt/homebrew/lib/libopencv_ximgproc.4.11.0.dylib
task7: /opt/homebrew/lib/libopencv_video.4.11.0.dylib
task7: /opt/homebrew/lib/libopencv_videoio.4.11.0.dylib
task7: /opt/homebrew/lib/libopencv_imgcodecs.4.11.0.dylib
task7: /opt/homebrew/lib/libopencv_objdetect.4.11.0.dylib
task7: /opt/homebrew/lib/libopencv_calib3d.4.11.0.dylib
task7: /opt/homebrew/lib/libopencv_dnn.4.11.0.dylib
task7: /opt/homebrew/lib/libopencv_features2d.4.11.0.dylib
task7: /opt/homebrew/lib/libopencv_flann.4.11.0.dylib
task7: /opt/homebrew/lib/libopencv_photo.4.11.0.dylib
task7: /opt/homebrew/lib/libopencv_imgproc.4.11.0.dylib
task7: /opt/homebrew/lib/libopencv_core.4.11.0.dylib
task7: CMakeFiles/task7.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --bold --progress-dir=/Users/aditshah/Desktop/PRCV/Project4/Task7/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable task7"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/task7.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/task7.dir/build: task7
.PHONY : CMakeFiles/task7.dir/build

CMakeFiles/task7.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/task7.dir/cmake_clean.cmake
.PHONY : CMakeFiles/task7.dir/clean

CMakeFiles/task7.dir/depend:
	cd /Users/aditshah/Desktop/PRCV/Project4/Task7/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /Users/aditshah/Desktop/PRCV/Project4/Task7 /Users/aditshah/Desktop/PRCV/Project4/Task7 /Users/aditshah/Desktop/PRCV/Project4/Task7/build /Users/aditshah/Desktop/PRCV/Project4/Task7/build /Users/aditshah/Desktop/PRCV/Project4/Task7/build/CMakeFiles/task7.dir/DependInfo.cmake "--color=$(COLOR)"
.PHONY : CMakeFiles/task7.dir/depend

