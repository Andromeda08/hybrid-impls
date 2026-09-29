#pragma once

#include <stdexcept>
#include <SDL3/SDL.h>

#include "VulkanRHI/VulkanRHI.hpp"

class Application
{
public:
    Application();

    void run();

private:
    void initWindow();

    SDL_Window*                mWindow  = nullptr;
    bool                       mRunning = false;

    std::unique_ptr<VulkanRHI> mVulkanRHI;
};

extern Application* gApplication;
extern SDL_Window*  gWindow;
