@echo off
"D:\\android.sdk\\cmake\\3.22.1\\bin\\cmake.exe" ^
  "-HD:\\Raine\\haixia\\OpenCV\\libcxx_helper" ^
  "-DCMAKE_SYSTEM_NAME=Android" ^
  "-DCMAKE_EXPORT_COMPILE_COMMANDS=ON" ^
  "-DCMAKE_SYSTEM_VERSION=33" ^
  "-DANDROID_PLATFORM=android-33" ^
  "-DANDROID_ABI=x86_64" ^
  "-DCMAKE_ANDROID_ARCH_ABI=x86_64" ^
  "-DANDROID_NDK=D:\\android.sdk\\ndk\\25.1.8937393" ^
  "-DCMAKE_ANDROID_NDK=D:\\android.sdk\\ndk\\25.1.8937393" ^
  "-DCMAKE_TOOLCHAIN_FILE=D:\\android.sdk\\ndk\\25.1.8937393\\build\\cmake\\android.toolchain.cmake" ^
  "-DCMAKE_MAKE_PROGRAM=D:\\android.sdk\\cmake\\3.22.1\\bin\\ninja.exe" ^
  "-DCMAKE_LIBRARY_OUTPUT_DIRECTORY=D:\\Raine\\haixia\\OpenCV\\build\\intermediates\\cxx\\Debug\\2n2n401f\\obj\\x86_64" ^
  "-DCMAKE_RUNTIME_OUTPUT_DIRECTORY=D:\\Raine\\haixia\\OpenCV\\build\\intermediates\\cxx\\Debug\\2n2n401f\\obj\\x86_64" ^
  "-DCMAKE_BUILD_TYPE=Debug" ^
  "-BD:\\Raine\\haixia\\OpenCV\\.cxx\\Debug\\2n2n401f\\x86_64" ^
  -GNinja ^
  "-DANDROID_STL=c++_shared"
