#include "Application.hpp"

Application* gApplication = nullptr;
SDL_Window*  gWindow      = nullptr;

Application::Application()
: mRunning(false)
{
    initWindow();

    mVulkanRHI = std::make_unique<VulkanRHI>(mWindow);

    SDL_ShowWindow(mWindow);
}

void Application::run()
{
    mRunning = true;
    while (mRunning)
    {
        // Input
        SDL_Event event;
        while (SDL_PollEvent(&event))
        {
            switch (event.type)
            {
                case SDL_EVENT_KEY_DOWN: {
                    const SDL_KeyboardEvent& keyboardEvent = event.key;
                    if (keyboardEvent.key == SDLK_ESCAPE)
                    {
                        mRunning = false;
                    }
                    break;
                }
                case SDL_EVENT_WINDOW_CLOSE_REQUESTED: {
                    mRunning = false;
                    break;
                }
                default: {
                    break;
                }
            }
        }

        const Frame frame = mVulkanRHI->beginFrame();
        auto* pCommandList = frame.pCommandList;

        auto* pImage = mVulkanRHI->getSwapchain()->getTexture(frame.acquiredIndex);
        Barrier()
            .addBarrier(pImage->getBarrier(ImageUsage::Clear))
            .insert(pCommandList);

        pCommandList->getHandle().clearColorImage(
            pImage->getHandle(),
            vk::ImageLayout::eGeneral,
            vk::ClearColorValue().setFloat32({ 0.1f, 0.1f, 0.5f, 1.0f }),
            pImage->getSubresourceRange());

        Barrier()
            .addBarrier(pImage->getBarrier(ImageUsage::PresentSrc))
            .insert(pCommandList);

        mVulkanRHI->endFrameAndPresent(frame);
    }
}

void Application::initWindow()
{
    if (!SDL_Init(SDL_INIT_VIDEO))
    {
        throw std::runtime_error("Failed to init SDL");
    }

    mWindow = SDL_CreateWindow("meow", 1920, 1080, SDL_WINDOW_VULKAN | SDL_WINDOW_HIGH_PIXEL_DENSITY | SDL_WINDOW_HIDDEN);
    if (!mWindow)
    {
        throw std::runtime_error("Failed to create Window");
    }

    gWindow = mWindow;
}
