# CMAKE generated file: DO NOT EDIT!
# Generated by "NMake Makefiles" Generator, CMake Version 3.16

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

!IF "$(OS)" == "Windows_NT"
NULL=
!ELSE
NULL=nul
!ENDIF
SHELL = cmd.exe

# The CMake executable.
CMAKE_COMMAND = "G:\Programs\CLion\CLion 2020.1\bin\cmake\win\bin\cmake.exe"

# The command to remove a file.
RM = "G:\Programs\CLion\CLion 2020.1\bin\cmake\win\bin\cmake.exe" -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = G:\Projects\CUDA\lab4

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = G:\Projects\CUDA\lab4\cmake-build-debug

# Include any dependencies generated for this target.
include CMakeFiles\lab4.dir\depend.make

# Include the progress variables for this target.
include CMakeFiles\lab4.dir\progress.make

# Include the compile flags for this target's objects.
include CMakeFiles\lab4.dir\flags.make

CMakeFiles\lab4.dir\main.cu.obj: CMakeFiles\lab4.dir\flags.make
CMakeFiles\lab4.dir\main.cu.obj: ..\main.cu
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=G:\Projects\CUDA\lab4\cmake-build-debug\CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CUDA object CMakeFiles/lab4.dir/main.cu.obj"
	G:\Programs\CUDA\Development\bin\nvcc.exe  $(CUDA_DEFINES) $(CUDA_INCLUDES) $(CUDA_FLAGS) -x cu -dc G:\Projects\CUDA\lab4\main.cu -o CMakeFiles\lab4.dir\main.cu.obj -Xcompiler=-FdCMakeFiles\lab4.dir\,-FS

CMakeFiles\lab4.dir\main.cu.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CUDA source to CMakeFiles/lab4.dir/main.cu.i"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_PREPROCESSED_SOURCE

CMakeFiles\lab4.dir\main.cu.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CUDA source to assembly CMakeFiles/lab4.dir/main.cu.s"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_ASSEMBLY_SOURCE

# Object files for target lab4
lab4_OBJECTS = \
"CMakeFiles\lab4.dir\main.cu.obj"

# External object files for target lab4
lab4_EXTERNAL_OBJECTS =

CMakeFiles\lab4.dir\cmake_device_link.obj: CMakeFiles\lab4.dir\main.cu.obj
CMakeFiles\lab4.dir\cmake_device_link.obj: CMakeFiles\lab4.dir\build.make
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=G:\Projects\CUDA\lab4\cmake-build-debug\CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CUDA device code CMakeFiles\lab4.dir\cmake_device_link.obj"
	G:\Programs\CUDA\Development\bin\nvcc.exe -D_WINDOWS -Xcompiler=" /GR /EHsc" -Xcompiler="-Zi -Ob0 -Od /RTC1" -Xcompiler=-MDd -Wno-deprecated-gpu-targets -shared -dlink $(lab4_OBJECTS) $(lab4_EXTERNAL_OBJECTS) -o CMakeFiles\lab4.dir\cmake_device_link.obj  kernel32.lib user32.lib gdi32.lib winspool.lib shell32.lib ole32.lib oleaut32.lib uuid.lib comdlg32.lib advapi32.lib  -Xcompiler=-FdG:\Projects\CUDA\lab4\cmake-build-debug\CMakeFiles\lab4.dir\,-FS

# Rule to build all files generated by this target.
CMakeFiles\lab4.dir\build: CMakeFiles\lab4.dir\cmake_device_link.obj

.PHONY : CMakeFiles\lab4.dir\build

# Object files for target lab4
lab4_OBJECTS = \
"CMakeFiles\lab4.dir\main.cu.obj"

# External object files for target lab4
lab4_EXTERNAL_OBJECTS =

lab4.exe: CMakeFiles\lab4.dir\main.cu.obj
lab4.exe: CMakeFiles\lab4.dir\build.make
lab4.exe: CMakeFiles\lab4.dir\cmake_device_link.obj
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=G:\Projects\CUDA\lab4\cmake-build-debug\CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Linking CUDA executable lab4.exe"
	"G:\Programs\CLion\CLion 2020.1\bin\cmake\win\bin\cmake.exe" -E vs_link_exe --intdir=CMakeFiles\lab4.dir --rc=C:\PROGRA~2\WI3CF2~1\10\bin\100183~1.0\x64\rc.exe --mt=C:\PROGRA~2\WI3CF2~1\10\bin\100183~1.0\x64\mt.exe --manifests  -- C:\PROGRA~2\MIB055~1\2019\COMMUN~1\VC\Tools\MSVC\1425~1.286\bin\Hostx64\x64\link.exe /nologo $(lab4_OBJECTS) $(lab4_EXTERNAL_OBJECTS) CMakeFiles\lab4.dir\cmake_device_link.obj @<<
 /out:lab4.exe /implib:lab4.lib /pdb:G:\Projects\CUDA\lab4\cmake-build-debug\lab4.pdb /version:0.0 /debug /INCREMENTAL /subsystem:console  kernel32.lib user32.lib gdi32.lib winspool.lib shell32.lib ole32.lib oleaut32.lib uuid.lib comdlg32.lib advapi32.lib  -LIBPATH:"G:/Programs/CUDA/Development/lib/x64" "cudadevrt.lib" "cudart_static.lib" 
<<

# Rule to build all files generated by this target.
CMakeFiles\lab4.dir\build: lab4.exe

.PHONY : CMakeFiles\lab4.dir\build

CMakeFiles\lab4.dir\clean:
	$(CMAKE_COMMAND) -P CMakeFiles\lab4.dir\cmake_clean.cmake
.PHONY : CMakeFiles\lab4.dir\clean

CMakeFiles\lab4.dir\depend:
	$(CMAKE_COMMAND) -E cmake_depends "NMake Makefiles" G:\Projects\CUDA\lab4 G:\Projects\CUDA\lab4 G:\Projects\CUDA\lab4\cmake-build-debug G:\Projects\CUDA\lab4\cmake-build-debug G:\Projects\CUDA\lab4\cmake-build-debug\CMakeFiles\lab4.dir\DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles\lab4.dir\depend

