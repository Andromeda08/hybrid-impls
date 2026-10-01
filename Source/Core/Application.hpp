#pragma once

#include <SDL3/SDL.h>

#include "Camera.hpp"
#include "Hair/HairGeometry.hpp"
#include "VulkanRHI/VulkanRHI.hpp"

class Application
{
public:
    Application();

    ~Application();

    void run();

private:
    void initWindow();
    void initImGui() const;

    void loadHairFile(const std::string& name);

    static bool uiWantCaptureInput();

    SDL_Window*                   mWindow  = nullptr;
    bool                          mRunning = false;

    std::unique_ptr<VulkanRHI>    mVulkanRHI;
    std::unique_ptr<CameraSystem> mCameraSystem;

    HairGeometry                  mHairGeometry;
    std::unique_ptr<Buffer>       mHairVertices;
    std::unique_ptr<Buffer>       mHairAttributes;
    std::unique_ptr<Buffer>       mStrandDescriptions;
};

extern Application* gApplication;
extern SDL_Window*  gWindow;
