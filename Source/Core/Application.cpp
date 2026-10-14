#include "Application.hpp"

#include <print>

#include <imgui.h>
#include <backends/imgui_impl_sdl3.h>
#include <backends/imgui_impl_vulkan.h>

#include "Configuration.hpp"
#include "Hair/CyLoader.hpp"


Application* gApplication = nullptr;
SDL_Window*  gWindow      = nullptr;

Application::Application()
{
    initWindow();

    mVulkanRHI = std::make_unique<VulkanRHI>(mWindow);

    mCameraSystem = std::make_unique<CameraSystem>(mVulkanRHI.get());
    mCameraSystem->addCamera<OrbitCamera>(115.0f);

    const auto [w, h] = mVulkanRHI->getRenderResolution();
    mCameraSystem->addCamera<FlyingCamera>(glm::ivec2 {w, h}, glm::vec3 { 0.0f, 8.0f, 15.0f });

    for (auto&& [i, depthBuffer] : std::views::enumerate(mDepthBuffers))
    {
        depthBuffer = Texture::Builder()
            .setSize(mVulkanRHI->getRenderResolution())
            .setFormat(vk::Format::eD32Sfloat)
            .setLabel(fmt::format("DepthBuffer#{}", i))
            .create(mVulkanRHI->getDevice());
    }

    loadHairFile("a_wWavy");
    initImGui();

    mProfiler = std::make_unique<GPUProfiler>(mVulkanRHI.get());
    mHairRendererParams = std::make_unique<HairRendererParams>(&mHairGeometry);
    mBaselinePass = std::make_unique<BaselinePass>(mHairRendererParams.get(), mVulkanRHI.get());
    mTwoPassHybrid = std::make_unique<TwoPassHybrid>(mHairRendererParams.get(), mVulkanRHI.get());

    SDL_ShowWindow(mWindow);
}

Application::~Application()
{
    mVulkanRHI->getDevice()->waitIdle();

    ImGui_ImplVulkan_Shutdown();
    ImGui_ImplSDL3_Shutdown();
    ImGui::DestroyContext();

    SDL_DestroyWindow(mWindow);
    SDL_Quit();
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
            ImGui_ImplSDL3_ProcessEvent(&event);
            if (!uiWantCaptureInput())
            {
                mCameraSystem->onEvent(event);
            }

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

        mCameraSystem->onUpdate(frame);

        auto* pImage = mVulkanRHI->getSwapchain()->getTexture(frame.acquiredIndex);
        Barrier()
            .addBarrier(pImage->getBarrier(ImageUsage::Clear))
            .insert(pCommandList);

        pCommandList->getHandle().clearColorImage(
            pImage->getHandle(),
            vk::ImageLayout::eGeneral,
            vk::ClearColorValue().setFloat32({ 0.0f, 0.005f, 0.01f, 1.0f }),
            pImage->getSubresourceRange());

        const auto hairBuffers = HairBuffers
        {
            .vertexBuffer            = mHairVertices->getAddress(),
            .attributesBuffer        = mHairAttributes->getAddress(),
            .strandDescriptionBuffer = mStrandDescriptions->getAddress(),
            .cameraBuffer            = mCameraSystem->getBuffer(frame.currentFrame)->getAddress(),
        };

        mProfiler->beginQuery(pCommandList, frame.currentFrame);
        {
            //mBaselinePass->render(frame, hairBuffers, mDepthBuffers[frame.currentFrame].get());
            mTwoPassHybrid->render(frame, hairBuffers);
        }
        mProfiler->endQuery(pCommandList, frame.currentFrame);

        // User Interface
        // =====================================
        Rendering()
            .setLabel("ImGui")
            .usingCommandList(pCommandList)
            .addAttachment(pImage, vk::AttachmentLoadOp::eLoad)
            .execute([&](const CommandList* pCmd) -> void
            {
                ImGui_ImplVulkan_NewFrame();
                ImGui_ImplSDL3_NewFrame();
                ImGui::NewFrame();
                {
                    ImGui::Begin("Benchmark");
                    {
                        ImGui::Text("Pipeline time: %f", mProfiler->lastGpuMS);
                        ImGui::Text("Primitives: %llu", mProfiler->lastMeshPrims);

                        ImGui::BeginDisabled(mProfiler->isCapturing());
                        if (ImGui::Button("Start Capture"))
                        {
                            mProfiler->startCapture();
                        }
                        ImGui::EndDisabled();

                        if (mProfiler->isCapturing())
                        {
                            ImGui::SameLine();
                            ImGui::Text("Progress: %llu / %llu", mProfiler->getCaptureFrame(), GPUProfiler::sCaptureFrames);
                        }
                    }
                    ImGui::End();

                    mCameraSystem->onDrawUI();
                    mHairRendererParams->onDrawUI();
                }
                ImGui::EndFrame();
                ImGui::Render();
                ImDrawData* drawData = ImGui::GetDrawData();
                ImGui_ImplVulkan_RenderDrawData(drawData, pCmd->getHandle());
            });

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

void Application::initImGui() const
{
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    const ImGuiIO& io = ImGui::GetIO();
    io.Fonts->AddFontFromFileTTF("GeistMono-Regular.ttf", 16.0f);
    ImGui::StyleColorsDark();

    [[maybe_unused]] auto bResult = ImGui_ImplSDL3_InitForVulkan(mWindow);
    assert(bResult);

    constexpr VkFormat format = VK_FORMAT_B8G8R8A8_UNORM;
    ImGui_ImplVulkan_InitInfo initInfo = {};
    initInfo.ApiVersion = VK_API_VERSION_1_4;
    initInfo.Instance = mVulkanRHI->getInstance()->getHandle();
    initInfo.PhysicalDevice = mVulkanRHI->getDevice()->getPhysicalDevice();
    initInfo.Device = mVulkanRHI->getDevice()->getHandle();
    initInfo.QueueFamily = mVulkanRHI->getDevice()->getGraphicsQueue().familyIndex;
    initInfo.Queue = mVulkanRHI->getDevice()->getGraphicsQueue().queue;
    initInfo.PipelineCache = VK_NULL_HANDLE;
    initInfo.DescriptorPool = VK_NULL_HANDLE;
    initInfo.DescriptorPoolSize = IMGUI_IMPL_VULKAN_MINIMUM_SAMPLED_IMAGE_POOL_SIZE;
    initInfo.MinImageCount = gFramesInFlight;
    initInfo.ImageCount = gFramesInFlight;
    initInfo.UseDynamicRendering = true;
    initInfo.PipelineInfoMain.PipelineRenderingCreateInfo = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO,
        .pNext = nullptr,
        .viewMask = {},
        .colorAttachmentCount = 1,
        .pColorAttachmentFormats = &format,
        .depthAttachmentFormat = VK_FORMAT_UNDEFINED,
        .stencilAttachmentFormat = VK_FORMAT_UNDEFINED,
    };

    bResult = ImGui_ImplVulkan_Init(&initInfo);
    assert(bResult);
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

bool Application::uiWantCaptureInput()
{
    const ImGuiIO& io = ImGui::GetIO();
    return io.WantCaptureMouse || io.WantCaptureKeyboard;
}
