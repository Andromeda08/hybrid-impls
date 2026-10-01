#include "VulkanRHI.hpp"

#include <fstream>
#include <print>
#include <vulkan/vulkan_format_traits.hpp>

#include "Core/Configuration.hpp"

VULKAN_HPP_DEFAULT_DISPATCH_LOADER_DYNAMIC_STORAGE;

Buffer::Buffer(const Builder& b, const std::shared_ptr<Device>& device)
: size(b.size)
, type(b.type)
, label(b.label)
, mDevice(device)
{
    const auto createInfo = vk::BufferCreateInfo()
        .setSize(size)
        .setUsage(getBufferUsageFlags(type));

    VmaAllocationCreateInfo allocationCreateInfo = {};
    allocationCreateInfo.usage = (type == BufferType::Staging || type == BufferType::Readback)
        ? VMA_MEMORY_USAGE_AUTO_PREFER_HOST
        : VMA_MEMORY_USAGE_AUTO;
    allocationCreateInfo.flags = getBufferMemoryFlags(type);

    const auto result = vmaCreateBuffer(
        mDevice->getAllocator(),
        &static_cast<const VkBufferCreateInfo&>(createInfo),
        &allocationCreateInfo,
        reinterpret_cast<VkBuffer*>(&mBuffer),
        &mAlloc,
        &mAllocInfo);

    if (result != VK_SUCCESS)
    {
        throw std::runtime_error(fmt::format("Failed to create Buffer: {}", string_VkResult(result)));
    }

    const auto addressInfo = vk::BufferDeviceAddressInfo().setBuffer(mBuffer);
    mAddress = mDevice->getHandle().getBufferAddress(&addressInfo);
}

vk::BufferUsageFlags Buffer::getBufferUsageFlags(const BufferType bufferType) noexcept
{
    using enum vk::BufferUsageFlagBits;
    vk::BufferUsageFlags result = eTransferSrc | eTransferDst | eShaderDeviceAddress;

    switch (bufferType)
    {
        case BufferType::Index:{
            result |= eIndexBuffer | eStorageBuffer;
            break;
        }
        case BufferType::Vertex:{
            result |= eVertexBuffer | eStorageBuffer;
            break;
        }
        case BufferType::Indirect:{
            result |= eIndirectBuffer;
            break;
        }
        case BufferType::Storage: {
            result |= eStorageBuffer;
            break;
        }
        case BufferType::Uniform: {
            result |= eUniformBuffer;
            break;
        }
        case BufferType::AccelerationStructure: {
            result |= eAccelerationStructureStorageKHR;
            break;
        }
        case BufferType::ShaderBindingTable: {
            result |= eShaderBindingTableKHR;
            break;
        }
        case BufferType::Staging: {
            break;
        }
        case BufferType::Readback: {
            result |= eStorageBuffer;
            break;
        }
        case BufferType::DescriptorHeap: {
            result |= eDescriptorHeapEXT;
        }
    }

    return result;
}

#pragma region "Texture"

std::unique_ptr<Texture> Texture::Builder::create(const std::shared_ptr<Device>& device) const
{
    return std::make_unique<Texture>(*this, device);
}

std::unique_ptr<Texture> Texture::Builder::wrap(const std::shared_ptr<Device>& device, const vk::Image image) const
{
    return std::make_unique<Texture>(*this, device, image);
}

Texture::Texture(const Builder& b, const std::shared_ptr<Device>& device)
: format(b.format)
, size(b.size)
, label(b.label)
, mDevice(device)
{
    assert(mDevice);

    createImage();
    mDevice->setLabel(mImage, b.label);

    createImageView();
    mDevice->setLabel(mImageView, fmt::format("{} View", b.label));
}

Texture::Texture(const Builder& b, const std::shared_ptr<Device>& device, const vk::Image image)
: format(b.format)
, size(b.size)
, label(b.label)
, mDevice(device)
, mIsWrapper(true)
, mImage(image)
{
    assert(mDevice);
    assert(mImage);

    mDevice->setLabel(mImage, b.label);

    createImageView();
    mDevice->setLabel(mImageView, fmt::format("{} View", b.label));
}

Texture::~Texture()
{
    mDevice->getHandle().destroy(mImageView);
    if (!mIsWrapper)
    {
        vmaDestroyImage(mDevice->getAllocator(), mImage, mAlloc);
    }
}

vk::ImageMemoryBarrier2 Texture::getBarrier(const ImageUsage dstUsage)
{
    const auto nextState = TrackedState::fromImageUsage(dstUsage);
    const auto [ newLayout, newAccess, newStage ] = nextState;

    const auto barrier = vk::ImageMemoryBarrier2()
        .setOldLayout(mState.layout)
        .setNewLayout(newLayout)
        .setSrcAccessMask(mState.accessMask)
        .setDstAccessMask(newAccess)
        .setSrcStageMask(mState.stageMask)
        .setDstStageMask(newStage)
        .setSrcQueueFamilyIndex(VK_QUEUE_FAMILY_IGNORED)
        .setDstQueueFamilyIndex(VK_QUEUE_FAMILY_IGNORED)
        .setImage(mImage)
        .setSubresourceRange(getSubresourceRange());

    mState = nextState;
    return barrier;
}

void Texture::createImage()
{
    using enum vk::ImageUsageFlagBits;
    const auto imageCreateInfo = vk::ImageCreateInfo()
        .setArrayLayers(1)
        .setExtent({ size.width, size.height, 1 })
        .setFormat(format)
        .setImageType(vk::ImageType::e2D)
        .setInitialLayout(vk::ImageLayout::eUndefined)
        .setMipLevels(1)
        .setSamples(vk::SampleCountFlagBits::e1)
        .setSharingMode(vk::SharingMode::eExclusive)
        .setTiling(vk::ImageTiling::eOptimal)
        .setUsage((vk::hasDepthComponent(format) ? eDepthStencilAttachment : eColorAttachment) |
            eTransferSrc | eTransferDst | eHostTransfer | eStorage | eSampled);

    constexpr VmaAllocationCreateInfo allocationCreateInfo = {
        .usage = VMA_MEMORY_USAGE_AUTO,
    };

    const auto result = vmaCreateImage(
        mDevice->getAllocator(),
        &static_cast<const VkImageCreateInfo&>(imageCreateInfo),
        &allocationCreateInfo,
        reinterpret_cast<VkImage*>(&mImage),
        &mAlloc,
        &mAllocInfo);

    if (result != VK_SUCCESS)
    {
        throw std::runtime_error(fmt::format("Failed to create Image: {}", string_VkResult(result)));
    }
}

void Texture::createImageView()
{
    mViewCreateInfo = vk::ImageViewCreateInfo()
        .setFormat(format)
        .setImage(mImage)
        .setSubresourceRange(getSubresourceRange())
        .setViewType(vk::ImageViewType::e2D);

    mImageView = mDevice->getHandle().createImageView(mViewCreateInfo);
}

vk::ImageSubresourceRange Texture::getSubresourceRange() const noexcept
{
    return {
        (vk::hasDepthComponent(format) ? vk::ImageAspectFlagBits::eDepth : vk::ImageAspectFlagBits::eColor),
        0, 1, 0, 1
    };
}

vk::ImageSubresourceLayers Texture::getSubresourceLayers() const noexcept
{
    return {
        (vk::hasDepthComponent(format) ? vk::ImageAspectFlagBits::eDepth : vk::ImageAspectFlagBits::eColor),
        0, 0, 1
    };
}

#pragma endregion

Swapchain::Swapchain(SDL_Window* pWindow, Instance* pInstance, const vk::Extent2D& size, const std::shared_ptr<Device>& device)
: mWindow(pWindow)
, mInstance(pInstance)
, mDevice(device)
{
    assert(mWindow && mInstance);
    if (!SDL_Vulkan_CreateSurface(mWindow, mInstance->getHandle(), nullptr, reinterpret_cast<VkSurfaceKHR*>(&mSurface)))
    {
        throw std::runtime_error("surface creation failed");
    }

    const auto surfaceInfo = vk::PhysicalDeviceSurfaceInfo2KHR().setSurface(mSurface);
    const auto surfaceCaps = mDevice->getPhysicalDevice().getSurfaceCapabilities2KHR(surfaceInfo).surfaceCapabilities;

    using enum vk::ImageUsageFlagBits;
    const auto swapchainCreateInfo = vk::SwapchainCreateInfoKHR()
        .setClipped(true)
        .setImageArrayLayers(1)
        .setImageSharingMode(vk::SharingMode::eExclusive)
        .setImageUsage(eColorAttachment | eTransferSrc | eTransferDst)
        .setOldSwapchain(nullptr)
        .setPQueueFamilyIndices(nullptr)
        .setQueueFamilyIndexCount(0)
        .setSurface(mSurface)
        .setPreTransform(surfaceCaps.currentTransform)
        .setMinImageCount(gFramesInFlight)
        .setImageExtent(size)
        .setImageFormat(vk::Format::eB8G8R8A8Unorm)
        .setImageColorSpace(vk::ColorSpaceKHR::eSrgbNonlinear)
        .setPresentMode(vk::PresentModeKHR::eMailbox);

    mSwapchain = mDevice->getHandle().createSwapchainKHR(swapchainCreateInfo);

    const auto images = mDevice->getHandle().getSwapchainImagesKHR(mSwapchain);
    mTextures.resize(images.size());
    for (auto&& [i, image] : std::views::enumerate(images))
    {
        mTextures[i] = Texture::Builder()
            .setFormat(vk::Format::eB8G8R8A8Unorm)
            .setSize(size)
            .setLabel(fmt::format("SwapchainImage-{}", i))
            .wrap(mDevice, image);
    }
}

#pragma region "Rendering"

Rendering& Rendering::setLabel(const std::string& value)
{
    mLabel = value;
    return *this;
}

Rendering& Rendering::setRenderArea(const vk::Extent2D& extent)
{
    mAutoRenderArea = false;
    mRenderArea = vk::Rect2D().setExtent(extent).setOffset({0, 0});
    return *this;
}

Rendering& Rendering::setRenderArea(const vk::Rect2D& rect)
{
    mAutoRenderArea = false;
    mRenderArea = rect;
    return *this;
}

Rendering& Rendering::addAttachment(
    Texture* pImage, const vk::AttachmentLoadOp loadOp, const vk::AttachmentStoreOp storeOp,
    const std::optional<vk::ClearValue>& clearValue, Texture* pResolve)
{
    if (mAutoRenderArea)
    {
        setRenderArea(pImage->size);
    }

    const bool isDepth     = vk::hasDepthComponent(pImage->format);
    const auto _clearValue = clearValue.value_or(isDepth
        ? vk::ClearValue().setDepthStencil({1.0f, 0})
        : vk::ClearValue().setColor({0.0f, 0.0f, 0.0f, 1.0f}));

    auto attachmentInfo = vk::RenderingAttachmentInfo()
        .setClearValue(_clearValue)
        .setImageLayout(vk::ImageLayout::eGeneral)
        .setImageView(pImage->getImageView())
        .setLoadOp(loadOp)
        .setStoreOp(pResolve ? vk::AttachmentStoreOp::eDontCare : storeOp);

    if (pResolve)
    {
        attachmentInfo
            .setResolveImageLayout(vk::ImageLayout::eGeneral)
            .setResolveImageView(pResolve->getImageView())
            .setResolveMode(isDepth
                ? vk::ResolveModeFlagBits::eSampleZero
                : vk::ResolveModeFlagBits::eAverage);

        mResolveImages.push_back(pResolve);
    }

    mImages.push_back(pImage);

    if (isDepth)
    {
        mDepthAttachmentInfo = attachmentInfo;
    }
    else
    {
        mAttachmentInfos.push_back(attachmentInfo);
    }

    return *this;
}

Barrier Rendering::getBarriers()
{
    auto barriers = Barrier();

    auto both = std::array{ std::span{mImages}, std::span{mResolveImages} };
    for (auto* pImage : both | std::views::join)
    {
        if (vk::hasDepthComponent(pImage->format))
        {
            barriers.addBarrier(pImage->getBarrier(ImageUsage::DepthAttachment));
        }
        else
        {
            barriers.addBarrier(pImage->getBarrier(ImageUsage::ColorAttachment));
        }
    }

    return barriers;
}

Rendering& Rendering::insertBarriers(const CommandList* pCommandList)
{
    getBarriers().insert(pCommandList);
    return *this;
}

Rendering& Rendering::setViewportScissor(const CommandList* pCommandList)
{
    const auto scissor  = mRenderArea;
    const auto viewport = vk::Viewport {
        static_cast<float>(mRenderArea.offset.x),
        static_cast<float>(mRenderArea.offset.y),
        static_cast<float>(mRenderArea.extent.width),
        static_cast<float>(mRenderArea.extent.height),
        0.0f,
        1.0f
    };
    pCommandList->setViewportScissor(viewport, scissor);
    return *this;
}

#pragma endregion

ShaderInfo ShaderInfo::fromFileName(const std::string& fileName)
{
    using enum vk::ShaderStageFlagBits;
    auto stage = eAll;

    // Resolve shader stage from standard file extensions
    #pragma region
    #define nbl_StageCase(Ext, Stage) if (std::string_view(substr) == Ext) { stage = Stage; }
    for (const auto substr : std::views::split(fileName, '.'))
    {
        nbl_StageCase("vert",   eVertex)
        nbl_StageCase("geom",   eGeometry)
        nbl_StageCase("tc",     eTessellationControl)
        nbl_StageCase("te",     eTessellationEvaluation)
        nbl_StageCase("frag",   eFragment)
        nbl_StageCase("comp",   eCompute)
        nbl_StageCase("mesh",   eMeshEXT)
        nbl_StageCase("task",   eTaskEXT)
        nbl_StageCase("rgen",   eRaygenKHR)
        nbl_StageCase("chit",  eClosestHitKHR)
        nbl_StageCase("miss",  eMissKHR)
        nbl_StageCase("rahit",  eAnyHitKHR)
        nbl_StageCase("rint",   eIntersectionKHR)
        nbl_StageCase("call",  eCallableKHR)
    }
    #undef nbl_StageCase
    #pragma endregion

    if (stage == eAll)
    {
        throw std::runtime_error("Failed to resolve shader stage from fileName");
    }

    return {
        .path       = Configuration::getShaderFilePath(fileName),
        .stage      = stage,
        .entryPoint = "main"
    };
}

#pragma region "PipelineCommon"

PipelineCommon& PipelineCommon::setLabel(const std::string& value)
{
    label = value;
    return *this;
}

PipelineCommon& PipelineCommon::addShader(const ShaderInfo& shaderInfo)
{
    shaders.push_back(shaderInfo);
    return *this;
}

PipelineCommon& PipelineCommon::addShader(const std::string& fileName)
{
    shaders.push_back(ShaderInfo::fromFileName(fileName));
    return *this;
}

#pragma endregion

#pragma region "GraphicsPS"

GraphicsPS& GraphicsPS::addAttachmentFormat(const vk::Format value)
{
    const bool isDepth   = vk::hasDepthComponent(value);
    const bool isStencil = vk::hasStencilComponent(value);

    if (isDepth)                depthFormat = value;
    if (isStencil)              stencilFormat = value;
    if (!isDepth && !isStencil) attachmentFormats.push_back(value);

    return *this;
}

GraphicsPS& GraphicsPS::addDefaultAttachmentState(const uint32_t count)
{
    using enum vk::ColorComponentFlagBits;
    for (uint32_t i = 0; i < count; i++)
    {
        constexpr auto state = vk::PipelineColorBlendAttachmentState()
            .setColorWriteMask(eR | eG | eB | eA)
            .setBlendEnable(false)
            .setSrcColorBlendFactor(vk::BlendFactor::eOne)
            .setDstColorBlendFactor(vk::BlendFactor::eZero)
            .setColorBlendOp(vk::BlendOp::eAdd)
            .setSrcAlphaBlendFactor(vk::BlendFactor::eOne)
            .setDstAlphaBlendFactor(vk::BlendFactor::eZero)
            .setAlphaBlendOp(vk::BlendOp::eAdd);
        blendStates.push_back(state);
    }
    return *this;
}

GraphicsPS& GraphicsPS::addAlphaAttachmentState(const uint32_t count)
{
    using enum vk::ColorComponentFlagBits;
    for (uint32_t i = 0; i < count; i++)
    {
        constexpr auto state = vk::PipelineColorBlendAttachmentState()
            .setColorWriteMask(eR | eG | eB | eA)
            .setBlendEnable(true)
            .setSrcColorBlendFactor(vk::BlendFactor::eSrcAlpha)
            .setDstColorBlendFactor(vk::BlendFactor::eOneMinusSrcAlpha)
            .setColorBlendOp(vk::BlendOp::eAdd)
            .setSrcAlphaBlendFactor(vk::BlendFactor::eOne)
            .setDstAlphaBlendFactor(vk::BlendFactor::eOneMinusSrcAlpha)
            .setAlphaBlendOp(vk::BlendOp::eAdd);
        blendStates.push_back(state);
    }
    return *this;
}

GraphicsPS& GraphicsPS::addColorBlendAttachmentState(const vk::PipelineColorBlendAttachmentState& blendState, const uint32_t count)
{
    for (uint32_t i = 0; i < count; i++)
    {
        blendStates.push_back(blendState);
    }
    return *this;
}

GraphicsPS& GraphicsPS::setCullMode(const vk::CullModeFlagBits cullMode)
{
    rasterizationState.setCullMode(cullMode);
    return *this;
}

GraphicsPS& GraphicsPS::setSampleCount(const vk::SampleCountFlagBits samples)
{
    multisampleState.setRasterizationSamples(samples);
    return *this;
}

GraphicsPS& GraphicsPS::setTopology(const vk::PrimitiveTopology topology)
{
    inputAssemblyState.setTopology(topology);
    return *this;
}

GraphicsPS& GraphicsPS::setWireframeMode(const bool value)
{
    rasterizationState.setPolygonMode(value ? vk::PolygonMode::eFill : vk::PolygonMode::eLine);
    return *this;
}

vk::PipelineInputAssemblyStateCreateInfo GraphicsPS::makeInputAssemblyState()
{
    return vk::PipelineInputAssemblyStateCreateInfo()
        .setTopology(vk::PrimitiveTopology::eTriangleList)
        .setPrimitiveRestartEnable(false)
        .setFlags({})
        .setPNext(nullptr);
}

vk::PipelineRasterizationStateCreateInfo GraphicsPS::makeRasterizationState()
{
    return vk::PipelineRasterizationStateCreateInfo()
        .setPolygonMode(vk::PolygonMode::eFill)
        .setCullMode(vk::CullModeFlagBits::eBack)
        .setFrontFace(vk::FrontFace::eCounterClockwise)
        .setDepthClampEnable(false)
        .setDepthBiasEnable(false)
        .setDepthBiasClamp(0.0f)
        .setDepthBiasSlopeFactor(0.0f)
        .setLineWidth(1.0f)
        .setRasterizerDiscardEnable(false)
        .setPNext(nullptr);
}

vk::PipelineMultisampleStateCreateInfo GraphicsPS::makeMultisampleState()
{
    return vk::PipelineMultisampleStateCreateInfo()
        .setRasterizationSamples(vk::SampleCountFlagBits::e1)
        .setSampleShadingEnable(false)
        .setPSampleMask(nullptr)
        .setAlphaToCoverageEnable(false)
        .setAlphaToOneEnable(false)
        .setPNext(nullptr);
}

vk::PipelineDepthStencilStateCreateInfo GraphicsPS::makeDepthStencilState()
{
    return vk::PipelineDepthStencilStateCreateInfo()
        .setDepthTestEnable(true)
        .setDepthWriteEnable(true)
        .setDepthCompareOp(vk::CompareOp::eLess)
        .setDepthBoundsTestEnable(false)
        .setStencilTestEnable(false);
}

vk::PipelineViewportStateCreateInfo GraphicsPS::makeViewportState()
{
    return vk::PipelineViewportStateCreateInfo()
        .setViewportCount(1)
        .setPViewports(nullptr)
        .setScissorCount(1)
        .setPScissors(nullptr)
        .setPNext(nullptr);
}

vk::PipelineDynamicStateCreateInfo GraphicsPS::makeDynamicState()
{
    return vk::PipelineDynamicStateCreateInfo()
        .setDynamicStateCount(0)
        .setPDynamicStates(nullptr)
        .setPNext(nullptr);
}

vk::PipelineColorBlendStateCreateInfo GraphicsPS::makeColorBlendState()
{
    return vk::PipelineColorBlendStateCreateInfo()
        .setLogicOp(vk::LogicOp::eClear)
        .setLogicOpEnable(false)
        .setAttachmentCount(0)
        .setPAttachments(nullptr)
        .setBlendConstants({0.0f, 0.0f, 0.0f, 0.0f})
        .setPNext(nullptr);
}

vk::PipelineVertexInputStateCreateInfo GraphicsPS::makeVertexInputState()
{
    return vk::PipelineVertexInputStateCreateInfo()
        .setVertexAttributeDescriptionCount(0)
        .setPVertexAttributeDescriptions(nullptr)
        .setVertexBindingDescriptionCount(0)
        .setPVertexBindingDescriptions(nullptr)
        .setPNext(nullptr);
}

#pragma endregion

#pragma region "Pipeline"

PipelineBase::PipelineBase(const PipelineCommon& common, const PipelineType type, const std::shared_ptr<Device>& device)
: mDevice(device)
, mType(type)
, mBindPoint(toBindPoint(type))
{
    label = common.label;
    std::println("Creating Pipeline: {}", label);
}

PipelineBase::~PipelineBase()
{
    mDevice->waitIdle();
    mDevice->getHandle().destroy(mPipeline);

    std::println("Destroyed Pipeline: {}", label);
}

const vk::Pipeline& PipelineBase::getHandle() const noexcept
{
    return mPipeline;
}

PipelineType PipelineBase::getType() const noexcept
{
    return mType;
}

const vk::PipelineBindPoint& PipelineBase::getBindPoint() const noexcept
{
    return mBindPoint;
}

vk::PipelineBindPoint PipelineBase::toBindPoint(const PipelineType type)
{
    using enum PipelineType;
    switch (type)
    {
        case Compute:    return vk::PipelineBindPoint::eCompute;
        default:         return vk::PipelineBindPoint::eGraphics;
    }
}

std::vector<char> PipelineBase::readShaderFile(const std::filesystem::path& filePath)
{
    std::ifstream file(filePath, std::ios::ate | std::ios::binary);
    if (!file.is_open())
    {
        throw std::runtime_error(fmt::format("Failed to open file: {}", filePath.string().c_str()));
    }

    const std::streamsize fileSize = file.tellg();
    std::vector<char>     buffer(fileSize);

    file.seekg(0);
    file.read(buffer.data(), fileSize);

    file.close();
    return buffer;
}

PipelineBase::LoadedShader PipelineBase::loadShader(const ShaderInfo& shaderInfo)
{
    LoadedShader result = {};
    result.shaderInfo   = shaderInfo;
    result.code         = readShaderFile(shaderInfo.path);
    return result;
}

#pragma endregion