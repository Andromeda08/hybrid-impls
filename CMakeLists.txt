cmake_minimum_required(VERSION 4.3)
project(thesispromax)

find_package(Vulkan REQUIRED)

set(CMAKE_CXX_STANDARD 23)

include(cmake/fetchSlang.cmake)

add_subdirectory(External/fmt EXCLUDE_FROM_ALL)
add_subdirectory(External/glm EXCLUDE_FROM_ALL)
add_subdirectory(External/sdl EXCLUDE_FROM_ALL)
add_subdirectory(External/vma EXCLUDE_FROM_ALL)

set(CY External/cy/cyCore.h External/cy/cyHairFile.h)

set("IMGUI_DIR" External/imgui)
set(IMGUI_FILES
    ${IMGUI_DIR}/backends/imgui_impl_vulkan.cpp ${IMGUI_DIR}/backends/imgui_impl_sdl3.cpp
    ${IMGUI_DIR}/imgui.cpp ${IMGUI_DIR}/imgui.h ${IMGUI_DIR}/imgui_draw.cpp
    ${IMGUI_DIR}/imgui_demo.cpp ${IMGUI_DIR}/imgui_tables.cpp ${IMGUI_DIR}/imgui_widgets.cpp
    ${IMGUI_DIR}/misc/cpp/imgui_stdlib.cpp
)

add_executable(exec
    ${CY} ${IMGUI_FILES}
    Source/main.cpp
    Source/Core/Application.hpp
    Source/Core/Application.cpp
    Source/Core/ShaderBuild.hpp
    Source/Core/ShaderBuild.cpp
    Source/Hair/CyLoader.hpp
    Source/Hair/CyLoader.cpp
    Source/Hair/HairGeometry.hpp
    Source/Hair/HairTypes.hpp
    Source/Core/Configuration.hpp
    Source/Core/Configuration.cpp
    Source/VulkanRHI/VulkanRHI.hpp
    Source/VulkanRHI/VulkanRHI.cpp
    Source/VulkanRHI/vma.impl.cpp
    Source/Hair/Baseline/BaselinePass.hpp
    Source/Core/Camera.hpp
    Source/Hair/HairBuffers.hpp
)

target_link_libraries(exec PUBLIC
    ${Vulkan_LIBRARIES}
    fmt
    glm::glm
    GPUOpen::VulkanMemoryAllocator
    SDL3::SDL3
    slang::slang
)

target_include_directories(exec PUBLIC
    ${Vulkan_INCLUDE_DIRS}
    Source
    External/cy
    External/fmt/include
    External/glm
    External/imgui
    External/imgui/backends
    External/imgui/misc/cpp
    External/vma/include
    External/sdl/include
)

target_compile_definitions(exec PUBLIC
    GLM_FORCE_DEPTH_ZERO_TO_ONE
    GLM_ENABLE_EXPERIMENTAL
    -DImTextureID=ImU64
    VULKAN_HPP_DISPATCH_LOADER_DYNAMIC=1
)