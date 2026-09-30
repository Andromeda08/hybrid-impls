#pragma once

#include <SDL3/SDL.h>

#include "Hair/HairGeometry.hpp"
#include "VulkanRHI/VulkanRHI.hpp"

class Application
{
public:
    Application();

    void run();

private:
    void initWindow();

    void loadHairFile(const std::string& name);

    SDL_Window*                 mWindow  = nullptr;
    bool                        mRunning = false;

    std::unique_ptr<VulkanRHI>  mVulkanRHI;

    HairGeometry                mHairGeometry;
    std::unique_ptr<Buffer>     mHairVertices;
    std::unique_ptr<Buffer>     mHairAttributes;
    std::unique_ptr<Buffer>     mStrandDescriptions;
};

extern Application* gApplication;
extern SDL_Window*  gWindow;
