include(FetchContent)

set(SLANG_VERSION "2026.13.1")

if(WIN32)
    set(_slang_pkg "windows-x86_64.zip")
elseif(APPLE)
    set(_slang_pkg "macos-aarch64.tar.gz")
else()
    set(_slang_pkg "linux-x86_64.tar.gz")
endif()

FetchContent_Declare(
    slang
    URL https://github.com/shader-slang/slang/releases/download/v${SLANG_VERSION}/slang-${SLANG_VERSION}-${_slang_pkg}
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE
)
FetchContent_MakeAvailable(slang)

add_library(slang::slang INTERFACE IMPORTED)
target_include_directories(slang::slang INTERFACE ${slang_SOURCE_DIR}/include)
target_link_directories(slang::slang INTERFACE ${slang_SOURCE_DIR}/lib)
target_link_libraries(slang::slang INTERFACE slang)
