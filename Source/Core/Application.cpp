#include "Application.hpp"

#include <print>

#include "Configuration.hpp"
#include "Hair/CyLoader.hpp"

Application* gApplication = nullptr;
SDL_Window*  gWindow      = nullptr;

Application::Application()
{
    initWindow();

    mVulkanRHI = std::make_unique<VulkanRHI>(mWindow);
    loadHairFile("a_wWavy");

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

void Application::loadHairFile(const std::string& name)
{
    mHairGeometry = CyLoader::loadModel(Configuration::getHairFilePath(fmt::format("{}.hair", name)));

    const uint64_t hairVerticesSize = mHairGeometry.getVertexCount() * sizeof(HairVertex);
    mHairVertices = Buffer::Builder()
        .setSize(hairVerticesSize)
        .setType(BufferType::Storage)
        .setLabel("HairVertices")
        .create(mVulkanRHI->getDevice());

    const uint64_t hairAttributesSize = mHairGeometry.getVertexCount() * sizeof(HairAttributes);
    mHairAttributes = Buffer::Builder()
        .setSize(hairAttributesSize)
        .setType(BufferType::Storage)
        .setLabel("HairAttributes")
        .create(mVulkanRHI->getDevice());

    const uint64_t strandDescriptionsSize = mHairGeometry.getStrandCount() * sizeof(HairStrandDesc);
    mStrandDescriptions = Buffer::Builder()
        .setSize(strandDescriptionsSize)
        .setType(BufferType::Storage)
        .setLabel("HairStrandDescriptions")
        .create(mVulkanRHI->getDevice());

    const auto staging = Buffer::Builder()
        .setSize(hairVerticesSize + hairAttributesSize + strandDescriptionsSize)
        .setType(BufferType::Staging)
        .setLabel("HairStaging")
        .create(mVulkanRHI->getDevice());
    staging->setData(mHairGeometry.vertices.data(), hairVerticesSize, 0);
    staging->setData(mHairGeometry.attributes.data(), hairAttributesSize, hairVerticesSize);
    staging->setData(mHairGeometry.strandDescriptions.data(), strandDescriptionsSize, hairVerticesSize + hairAttributesSize);

    mVulkanRHI->getCommandQueue()->immediate([&](const CommandList* pCommandList) -> void
    {
        const auto vertices   = vk::BufferCopy2 { 0, 0, hairVerticesSize };
        const auto attributes = vk::BufferCopy2 { hairVerticesSize, 0, hairAttributesSize };
        const auto strandDesc = vk::BufferCopy2 { hairVerticesSize + hairAttributesSize, 0, strandDescriptionsSize };

        const auto c0 = vk::CopyBufferInfo2().setSrcBuffer(staging->getHandle()).setDstBuffer(mHairVertices->getHandle()).setRegions(vertices);
        pCommandList->getHandle().copyBuffer2(c0);

        const auto c1 = vk::CopyBufferInfo2().setSrcBuffer(staging->getHandle()).setDstBuffer(mHairAttributes->getHandle()).setRegions(attributes);
        pCommandList->getHandle().copyBuffer2(c1);

        const auto c2 = vk::CopyBufferInfo2().setSrcBuffer(staging->getHandle()).setDstBuffer(mStrandDescriptions->getHandle()).setRegions(strandDesc);
        pCommandList->getHandle().copyBuffer2(c2);
    });

    std::println("{}", fmt::format(
        "Loaded Hair model: {}\n\t- Vertices: {}\n\t- Strands: {}",
        name, mHairGeometry.getVertexCount(), mHairGeometry.getStrandCount()));
}
