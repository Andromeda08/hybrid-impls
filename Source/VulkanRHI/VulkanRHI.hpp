#pragma once

#include <filesystem>
#include <functional>
#include <limits>
#include <map>
#include <memory>
#include <ranges>
#include <set>
#include <stdexcept>
#include <vector>
#include <vk_mem_alloc.h>
#include <fmt/format.h>
#include <SDL3/SDL_vulkan.h>
#include <vulkan/vk_enum_string_helper.h>
#include <vulkan/vulkan.hpp>

constexpr uint32_t gFramesInFlight = 3;

template <class T>
using PerFrameArray = std::array<T, gFramesInFlight>;

template <typename T>
concept VertexInput = requires (T t, uint32_t u, uint32_t b) {
    { T::getAttributes(u, b) } -> std::same_as<std::vector<vk::VertexInputAttributeDescription>>;
    { T::getBinding(u)       } -> std::same_as<vk::VertexInputBindingDescription>;
    { T::getAttributeCount() } -> std::same_as<uint32_t>;
};

template <typename T>
concept IndexType = std::is_same_v<T, uint16_t> || std::is_same_v<T, uint32_t>;

template <class Integral>
static constexpr Integral alignUp(Integral x, size_t a) noexcept
{
    return Integral((x + (Integral(a) - 1)) & ~Integral(a - 1));
}

class Instance final
{
public:
    explicit Instance()
    {
        constexpr auto applicationInfo = vk::ApplicationInfo().setApiVersion(vk::ApiVersion14);

        std::vector<const char*> extensions = {
            vk::EXTDebugUtilsExtensionName,
            vk::KHRSurfaceExtensionName,
            vk::KHRGetSurfaceCapabilities2ExtensionName,
            vk::KHRSurfaceMaintenance1ExtensionName,
        };

        uint32_t extensionCount;
        const auto sdlExt = SDL_Vulkan_GetInstanceExtensions(&extensionCount);
        for (uint32_t i = 0; i < extensionCount; i++)
        {
            extensions.push_back(sdlExt[i]);
        }

        const auto instanceCreateInfo = vk::InstanceCreateInfo()
            .setEnabledExtensionCount(static_cast<uint32_t>(extensions.size()))
            .setPpEnabledExtensionNames(extensions.data())
            .setPApplicationInfo(&applicationInfo);

        mInstance = vk::createInstance(instanceCreateInfo);
    }

    ~Instance()
    {
        if (mInstance)
        {
            mInstance.destroy();
        }
    }

    [[nodiscard]] vk::Instance getHandle() const noexcept
    {
        return mInstance;
    }

private:
    vk::Instance mInstance;
};

enum class QueueType
{
    Graphics,
    Compute,
    Transfer,
};

struct DeviceQueue
{
    vk::Queue  queue       {};
    uint32_t   familyIndex {};
    uint32_t   queueIndex  {};
    QueueType  queueType   {};
};

class Device final
{
public:
    explicit Device(Instance* pInstance)
    : mInstance(pInstance)
    {
        assert(mInstance);
        mPhysicalDevice = mInstance->getHandle().enumeratePhysicalDevices()[0];

        #pragma region "Extensions"
        auto core11 = vk::PhysicalDeviceVulkan11Features()
            .setShaderDrawParameters(true);
        auto core12 = vk::PhysicalDeviceVulkan12Features()
            .setShaderSampledImageArrayNonUniformIndexing(true)
            .setBufferDeviceAddress(true)
            .setDescriptorIndexing(true)
            .setScalarBlockLayout(true)
            .setShaderInt8(true)
            .setTimelineSemaphore(true)
            .setHostQueryReset(true)
            .setScalarBlockLayout(true)
            .setDrawIndirectCount(true)
            .setPNext(&core11);
        auto core13 = vk::PhysicalDeviceVulkan13Features()
            .setShaderDemoteToHelperInvocation(true)
            .setMaintenance4(true)
            .setDynamicRendering(true)
            .setSynchronization2(true)
            .setInlineUniformBlock(true)
            .setPNext(&core12);
        auto core14 = vk::PhysicalDeviceVulkan14Features()
            .setPushDescriptor(true)
            .setHostImageCopy(true)
            .setMaintenance5(true)
            .setMaintenance6(true)
            .setPNext(&core13);
        auto meshShader = vk::PhysicalDeviceMeshShaderFeaturesEXT()
            .setMeshShader(true)
            .setTaskShader(true)
            .setMeshShaderQueries(true)
            .setPNext(&core14);
        auto scmt1 = vk::PhysicalDeviceSwapchainMaintenance1FeaturesKHR()
            .setSwapchainMaintenance1(true)
            .setPNext(&meshShader);
        auto dheap = vk::PhysicalDeviceDescriptorHeapFeaturesEXT()
            .setDescriptorHeap(true)
            .setPNext(&scmt1);
        auto unified = vk::PhysicalDeviceUnifiedImageLayoutsFeaturesKHR()
            .setUnifiedImageLayouts(true)
            .setPNext(&dheap);
        auto deviceFault = vk::PhysicalDeviceFaultFeaturesEXT()
            .setDeviceFault(true)
            .setPNext(&unified);
        const auto features = vk::PhysicalDeviceFeatures()
            .setMultiDrawIndirect(true)
            .setFragmentStoresAndAtomics(true)
            .setDrawIndirectFirstInstance(true)
            .setFillModeNonSolid(false)
            .setSamplerAnisotropy(true)
            .setSampleRateShading(true)
            .setShaderInt64(true)
            .setTessellationShader(true)
            .setGeometryShader(true);
        #pragma endregion

        mExtensionNames = {
            vk::KHRDeferredHostOperationsExtensionName,
            vk::KHRSwapchainExtensionName,
            vk::KHRSwapchainMaintenance1ExtensionName,
            vk::KHRUnifiedImageLayoutsExtensionName,
            vk::EXTMeshShaderExtensionName,
            vk::EXTDeviceFaultExtensionName,
            vk::EXTDescriptorHeapExtensionName,
        };

        constexpr float queuePriority = 1.0f;
        std::set<uint32_t> uniqueQueueFamilies;

        const auto graphicsQueue = findQueueFamily(vk::QueueFlagBits::eGraphics);
        if (!graphicsQueue.has_value())
        {
            throw std::runtime_error("No graphics queue");
        }

        uniqueQueueFamilies.insert(graphicsQueue->second);

        std::vector<vk::DeviceQueueCreateInfo> queueCreateInfos;
        for (const uint32_t familyIndex : uniqueQueueFamilies)
        {
            const auto queueCreateInfo = vk::DeviceQueueCreateInfo()
                .setQueueFamilyIndex(familyIndex)
                .setQueueCount(1)
                .setPQueuePriorities(&queuePriority);
            queueCreateInfos.push_back(queueCreateInfo);
        }

        auto createInfo = vk::DeviceCreateInfo()
            .setEnabledExtensionCount(static_cast<uint32_t>(mExtensionNames.size()))
            .setPpEnabledExtensionNames(mExtensionNames.data())
            .setQueueCreateInfoCount(queueCreateInfos.size())
            .setPQueueCreateInfos(queueCreateInfos.data())
            .setPEnabledFeatures(&features)
            .setPNext(&deviceFault);

        mDevice = mPhysicalDevice.createDevice(createInfo);

        mGraphicsQueue = {
            .queue       = mDevice.getQueue(graphicsQueue->second, 0),
            .familyIndex = graphicsQueue->second,
            .queueIndex  = 0,
            .queueType   = QueueType::Graphics,
        };

        const VmaAllocatorCreateInfo allocatorCreateInfo = {
            .flags = VMA_ALLOCATOR_CREATE_BUFFER_DEVICE_ADDRESS_BIT,
            .physicalDevice = mPhysicalDevice,
            .device = mDevice,
            .instance = mInstance->getHandle(),
            .vulkanApiVersion = vk::ApiVersion14,
        };

        const auto result = vmaCreateAllocator(&allocatorCreateInfo, &mAllocator);
        assert(result == VK_SUCCESS);
    }

    ~Device()
    {
        if (mDevice)
        {
            mDevice.waitIdle();
            vmaDestroyAllocator(mAllocator);
            mDevice.destroy();
        }
    }

    void waitIdle() const
    {
        mDevice.waitIdle();
    }

    [[nodiscard]] vk::PhysicalDevice getPhysicalDevice() const noexcept
    {
        return mPhysicalDevice;
    }

    [[nodiscard]] vk::Device getHandle() const noexcept
    {
        return mDevice;
    }

    [[nodiscard]] VmaAllocator getAllocator() const noexcept
    {
        return mAllocator;
    }

    [[nodiscard]] const DeviceQueue& getGraphicsQueue() const noexcept
    {
        return mGraphicsQueue;
    }

    template <class T>
    requires vk::isVulkanHandleType<T>::value
    void setLabel(const T& handle, const std::optional<std::string>& label = std::nullopt) const
    {
        if (handle == VK_NULL_HANDLE)
        {
            return;
        }

        // ReSharper disable once CppFunctionalStyleCast
        const auto        objectHandle = uint64_t(static_cast<T::CType>(handle));
        const std::string objectName   = label.value_or(fmt::format("{} ({:#x})", vk::to_string(T::objectType), objectHandle));

        const auto objectNameInfo = vk::DebugUtilsObjectNameInfoEXT()
            .setPObjectName(objectName.c_str())
            .setObjectHandle(objectHandle)
            .setObjectType(T::objectType);

        mDevice.setDebugUtilsObjectNameEXT(objectNameInfo);
    }

private:
    [[nodiscard]] auto findQueueFamily(const vk::QueueFlags requiredFlags, const vk::QueueFlags excludedFlags = {}) const noexcept
    -> std::optional<std::pair<vk::QueueFamilyProperties, uint32_t>>
    {
        for (const std::vector queueFamilies = mPhysicalDevice.getQueueFamilyProperties2();
             auto&& [familyIndex, qp2] : std::views::enumerate(queueFamilies))
        {
            const auto& properties = qp2.queueFamilyProperties;
            if ((properties.queueCount > 0) && (properties.queueFlags & requiredFlags) && !(properties.queueFlags & excludedFlags))
            {
                return std::make_optional(std::make_pair(properties, static_cast<uint32_t>(familyIndex)));
            }
        }
        return std::nullopt;
    }

    Instance*                   mInstance;
    vk::PhysicalDevice          mPhysicalDevice;
    vk::Device                  mDevice;
    VmaAllocator                mAllocator {};
    std::vector<const char*>    mExtensionNames;
    DeviceQueue                 mGraphicsQueue;
};

enum class BufferUsage
{
    Compute,
    Compute_Read,
    Compute_Write,
    Fragment,
    Fragment_ReadWrite,
    AS_BuildUpdate,
    AS_BuildInput,
    AS_Traverse,
    SBT_Read,
    TransferSrc,
    TransferDst,
    DrawIndirect,
    ComputeIndirect,
    Vertex,
    Index,
    StorageRead,
    Host_Read,
    Host_Write,
    All,
};

enum class BufferType
{
    Index,
    Vertex,
    Indirect,
    Storage,
    Uniform,
    AccelerationStructure,
    ShaderBindingTable,
    Staging,
    Readback,
    DescriptorHeap,
};

class Buffer final
{
public:
    struct Builder
    {
        uint64_t    size  = 0;
        BufferType  type  = BufferType::Storage;
        std::string label = {};

        Builder& setSize(const uint64_t& value)
        {
            size = value;
            return *this;
        }

        Builder& setType(const BufferType& value)
        {
            type = value;
            return *this;
        }

        Builder& setLabel(const std::string& value)
        {
            label = value;
            return *this;
        }

        [[nodiscard]] std::unique_ptr<Buffer> create(const std::shared_ptr<Device>& device) const
        {
            return std::make_unique<Buffer>(*this, device);
        }
    };

    Buffer(const Builder& b, const std::shared_ptr<Device>& device);

    ~Buffer()
    {
        vmaDestroyBuffer(mDevice->getAllocator(), mBuffer, mAlloc);
    }

    void setData(const void* pData, const uint64_t dataSize, const uint64_t offset = 0) const
    {
        vmaCopyMemoryToAllocation(mDevice->getAllocator(), pData, mAlloc, offset, dataSize);
    }

    [[nodiscard]] void* getMapped() const noexcept
    {
        return mAllocInfo.pMappedData;
    }

    [[nodiscard]] vk::DeviceAddress getAddress() const noexcept
    {
        return mAddress;
    }

    [[nodiscard]] vk::Buffer getHandle() const noexcept
    {
        return mBuffer;
    }

    [[nodiscard]] vk::BufferMemoryBarrier2 getBarrier(const BufferUsage srcUsage, const BufferUsage dstUsage) const noexcept
    {
        const auto [ srcAccess, srcStage ] = getBarrierFlagsForBufferUsage(srcUsage);
        const auto [ dstAccess, dstStage ] = getBarrierFlagsForBufferUsage(dstUsage);
        return vk::BufferMemoryBarrier2()
            .setBuffer(mBuffer)
            .setSize(VK_WHOLE_SIZE)
            .setSrcAccessMask(srcAccess)
            .setDstAccessMask(dstAccess)
            .setSrcStageMask(srcStage)
            .setDstStageMask(dstStage);
    }

    const uint64_t    size;
    const BufferType  type;
    const std::string label;

private:
    [[nodiscard]] static vk::BufferUsageFlags getBufferUsageFlags(BufferType bufferType) noexcept;

    [[nodiscard]] static constexpr int32_t getBufferMemoryFlags(const BufferType bufferType)
    {
        switch (bufferType)
        {
            case BufferType::ShaderBindingTable:
            case BufferType::Staging:
            case BufferType::Uniform:
            case BufferType::DescriptorHeap:
                return VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT
                       | VMA_ALLOCATION_CREATE_HOST_ACCESS_ALLOW_TRANSFER_INSTEAD_BIT
                       | VMA_ALLOCATION_CREATE_MAPPED_BIT;
            case BufferType::Readback:
                return VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT
                     | VMA_ALLOCATION_CREATE_MAPPED_BIT;
            default:
                return 0;
        }
    }

    [[nodiscard]] static constexpr std::pair<vk::AccessFlags2, vk::PipelineStageFlags2> getBarrierFlagsForBufferUsage(const BufferUsage bufferUsage)
    {
        using A = vk::AccessFlagBits2;
        using S = vk::PipelineStageFlagBits2;

        using enum BufferUsage;
        switch (bufferUsage)
        {
            case Compute: {
                return { A::eShaderRead | A::eShaderWrite, S::eComputeShader | S::eMeshShaderEXT | S::eTaskShaderEXT };
            }
            case Compute_Read: {
                return { A::eShaderRead, S::eComputeShader | S::eMeshShaderEXT | S::eTaskShaderEXT };
            }
            case Compute_Write: {
                return { A::eShaderWrite, S::eComputeShader | S::eMeshShaderEXT | S::eTaskShaderEXT };
            }
            case Fragment: {
                return { A::eShaderRead, S::eFragmentShader};
            }
            case Fragment_ReadWrite: {
                return { A::eShaderRead | A::eShaderWrite, S::eFragmentShader};
            }
            case AS_BuildUpdate: {
                return { A::eAccelerationStructureWriteKHR, S::eAccelerationStructureBuildKHR | S::eAccelerationStructureCopyKHR };
            }
            case AS_BuildInput: {
                return { A::eShaderRead, S::eAccelerationStructureBuildKHR };
            }
            case AS_Traverse: {
                return { A::eAccelerationStructureReadKHR, S::eAllGraphics | S::eComputeShader | S::eRayTracingShaderKHR | S::eAccelerationStructureBuildKHR };
            }
            case SBT_Read: {
                return { A::eShaderBindingTableReadKHR, S::eAllGraphics };
            }
            case TransferSrc: {
                return { A::eTransferRead, S::eAllTransfer };
            }
            case TransferDst: {
                return { A::eTransferWrite, S::eAllTransfer };
            }
            case DrawIndirect: {
                return { A::eIndirectCommandRead, S::eDrawIndirect };
            }
            case ComputeIndirect: {
                return { A::eIndirectCommandRead, S::eComputeShader };
            }
            case Vertex: {
                return { A::eVertexAttributeRead, S::eVertexInput };
            }
            case Index: {
                return { A::eIndexRead, S::eVertexInput };
            }
            case StorageRead: {
                return { A::eShaderRead | A::eShaderStorageRead, S::eAllGraphics };
            }
            case Host_Read: {
                return { A::eHostRead, S::eHost };
            }
            case Host_Write: {
                return { A::eHostWrite, S::eHost };
            }
            case All: {
                return { A::eShaderWrite | A::eShaderRead | A::eTransferRead | A::eTransferWrite, S::eAllCommands };
            }
        }

        std::unreachable();
    }

    std::shared_ptr<Device> mDevice;
    vk::Buffer              mBuffer;
    vk::DeviceAddress       mAddress;
    VmaAllocation           mAlloc {};
    VmaAllocationInfo       mAllocInfo {};
};

enum class ImageUsage
{
    Undefined,
    ColorAttachment,
    DepthAttachment,
    Clear,
    BlitSrc,
    BlitDst,
    General,
    ShaderReadOnly,
    StorageImage,
    TransferSrc,
    TransferDst,
    PresentSrc,
};

class Texture final
{
public:
    struct Builder
    {
        vk::Format   format = vk::Format::eR32G32B32A32Sfloat;
        vk::Extent2D size   = { 512, 512 };
        std::string  label  = {};

        Builder& setFormat(const vk::Format value)
        {
            format = value;
            return *this;
        }

        Builder& setSize(const vk::Extent2D value)
        {
            size = value;
            return *this;
        }

        Builder& setLabel(const std::string& value)
        {
            label = value;
            return *this;
        }

        [[nodiscard]] std::unique_ptr<Texture> create(const std::shared_ptr<Device>& device) const;

        [[nodiscard]] std::unique_ptr<Texture> wrap(const std::shared_ptr<Device>& device, vk::Image image) const;
    };

    Texture(const Builder& b, const std::shared_ptr<Device>& device);

    Texture(const Builder& b, const std::shared_ptr<Device>& device, vk::Image image);

    ~Texture();

    const vk::Format    format;
    const vk::Extent2D  size;
    const std::string   label;

    [[nodiscard]] vk::ImageMemoryBarrier2 getBarrier(ImageUsage dstUsage);

    [[nodiscard]] vk::Image getHandle() const noexcept
    {
        return mImage;
    }

    [[nodiscard]] vk::ImageView getImageView() const noexcept
    {
        return mImageView;
    }

    [[nodiscard]] vk::ImageSubresourceRange getSubresourceRange() const noexcept;

    [[nodiscard]] vk::ImageSubresourceLayers getSubresourceLayers() const noexcept;

private:
    friend class DescriptorHeap;

    [[nodiscard]] const vk::ImageViewCreateInfo& getImageViewCreateInfo() const noexcept
    {
        return mViewCreateInfo;
    }

    void createImage();

    void createImageView();

    struct TrackedState
    {
        vk::ImageLayout         layout     = vk::ImageLayout::eUndefined;
        vk::AccessFlags2        accessMask = vk::AccessFlagBits2::eNone;
        vk::PipelineStageFlags2 stageMask  = vk::PipelineStageFlagBits2::eNone;

        [[nodiscard]] static TrackedState fromImageUsage(const ImageUsage usage) noexcept
        {
            using L = vk::ImageLayout;
            using A = vk::AccessFlagBits2;
            using S = vk::PipelineStageFlagBits2;

            using enum ImageUsage;
            switch (usage)
            {
                case Undefined: {
                    return {
                        .layout     = L::eUndefined,
                        .accessMask = A::eNone,
                        .stageMask  = S::eNone,
                    };
                }
                case ColorAttachment: {
                    return {
                        .layout     = L::eGeneral,
                        .accessMask = A::eColorAttachmentRead | A::eColorAttachmentWrite,
                        .stageMask  = S::eColorAttachmentOutput,
                    };
                }
                case DepthAttachment: {
                    return {
                        .layout     = L::eGeneral,
                        .accessMask = A::eDepthStencilAttachmentRead | A::eDepthStencilAttachmentWrite,
                        .stageMask  = S::eColorAttachmentOutput | S::eEarlyFragmentTests | S::eLateFragmentTests,
                    };
                }
                case Clear: {
                    return {
                        .layout     = L::eGeneral,
                        .accessMask = A::eTransferWrite,
                        .stageMask  = S::eClear,
                    };
                }
                case BlitSrc: {
                    return {
                        .layout     = L::eGeneral,
                        .accessMask = A::eTransferRead,
                        .stageMask  = S::eBlit,
                    };
                }
                case BlitDst: {
                    return {
                        .layout     = L::eGeneral,
                        .accessMask = A::eTransferWrite,
                        .stageMask  = S::eBlit,
                    };
                }
                case General: {
                    return {
                        .layout     = L::eGeneral,
                        .accessMask = A::eMemoryRead | A::eMemoryWrite,
                        .stageMask = S::eAllCommands,
                    };
                }
                case ShaderReadOnly: {
                    return {
                        .layout     = L::eGeneral,
                        .accessMask = A::eShaderRead | A::eShaderStorageRead | A::eShaderSampledRead,
                        .stageMask  = S::eAllCommands,
                    };
                }
                case StorageImage: {
                    return {
                        .layout     = L::eGeneral,
                        .accessMask = A::eShaderRead | A::eShaderWrite | A::eShaderStorageRead | A::eShaderStorageWrite,
                        .stageMask  = S::eAllCommands,
                    };
                }
                case TransferSrc: {
                    return {
                        .layout     = L::eGeneral,
                        .accessMask = A::eTransferRead,
                        .stageMask  = S::eAllTransfer,
                    };
                }
                case TransferDst: {
                    return {
                        .layout     = L::eGeneral,
                        .accessMask = A::eTransferWrite,
                        .stageMask  = S::eAllTransfer,
                    };
                }
                case PresentSrc: {
                    return {
                        .layout     = L::ePresentSrcKHR,
                        .accessMask = A::eNone,
                        .stageMask  = S::eBottomOfPipe,
                    };
                }
            }

            std::unreachable();
        }
    } mState;

    std::shared_ptr<Device> mDevice;
    const bool              mIsWrapper = false;

    vk::Image               mImage;
    vk::ImageViewCreateInfo mViewCreateInfo;
    vk::ImageView           mImageView;

    VmaAllocation           mAlloc {};
    VmaAllocationInfo       mAllocInfo {};
};

class Swapchain final
{
public:
    Swapchain(SDL_Window* pWindow, Instance* pInstance, const vk::Extent2D& size, const std::shared_ptr<Device>& device);

    ~Swapchain()
    {
        mTextures.clear();
        mDevice->getHandle().destroy(mSwapchain);
        mInstance->getHandle().destroy(mSurface);
    }

    [[nodiscard]] vk::SwapchainKHR getHandle() const noexcept
    {
        return mSwapchain;
    }

    [[nodiscard]] Texture* getTexture(const uint64_t i) const
    {
        return mTextures[i].get();
    };

private:
    SDL_Window*             mWindow;
    Instance*               mInstance;
    std::shared_ptr<Device> mDevice;

    vk::SurfaceKHR          mSurface;
    vk::SwapchainKHR        mSwapchain;

    std::vector<std::unique_ptr<Texture>> mTextures;
};

#pragma region "Pipelines"

struct ShaderInfo
{
    std::filesystem::path   path;
    vk::ShaderStageFlagBits stage;
    const char*             entryPoint = "main";

    /**
     * From the specified shader file resolve the stage using standard
     * engine file extensions and return a ShaderInfo struct.
     */
    [[nodiscard]] static ShaderInfo fromFileName(const std::string& fileName);
};

enum class PipelineType : uint8_t
{
    Unknown    = 0,
    Compute    = 1,
    Graphics   = 2,
};

/**
 * Builder for common options between all pipeline types.
 */
struct PipelineCommon
{
    // Debug Label
    std::string label;

    PipelineCommon& setLabel(const std::string& value);

    // Collection of shaders used by the pipeline.
    std::vector<ShaderInfo> shaders;

    // Add shader by explicit ShaderInfo
    PipelineCommon& addShader(const ShaderInfo& shaderInfo);

    // Add shader by file name and rely on automatic info resolution.
    PipelineCommon& addShader(const std::string& fileName);
};

/**
 * Builder for graphics pipeline state.
 */
struct GraphicsPS
{
    /**
     * Vertex Input Descriptions
     */
    std::vector<vk::VertexInputAttributeDescription> vtxInputAttributes;
    std::vector<vk::VertexInputBindingDescription>   vtxInputBindings;

    /**
     * Add a vertex input to the pipeline. They must be added in order of declaration,
     * the builder internally tracks the bindings & locations.
     * @tparam VertexT Vertex that satisfies the VertexInput concept.
     */
    template <VertexInput VertexT>
    GraphicsPS& addVertexType()
    {
        vtxInputAttributes.append_range(VertexT::getAttributes(vtxInputAttributes.size(), vtxInputBindings.size()));
        vtxInputBindings.push_back(VertexT::getBinding(vtxInputBindings.size()));
        return *this;
    }

    /**
     * Attachment formats and blend state
     */
    std::vector<vk::PipelineColorBlendAttachmentState>  blendStates         = {};
    std::vector<vk::Format>                             attachmentFormats   = {};
    vk::Format                                          depthFormat         = vk::Format::eUndefined;
    vk::Format                                          stencilFormat       = vk::Format::eUndefined;

    // Add an attachment format to the pipeline, color ones must be in order while depth and stencil are resolved automatically.
    GraphicsPS& addAttachmentFormat(vk::Format value);

    // Add a default color attachment blend state "count" times.
    GraphicsPS& addDefaultAttachmentState(uint32_t count = 1);

    // Add an alpha blending color attachment blend state "count" times.
    GraphicsPS& addAlphaAttachmentState(uint32_t count = 1);

    // Add a custom blend state "count" times.
    GraphicsPS& addColorBlendAttachmentState(const vk::PipelineColorBlendAttachmentState& blendState, uint32_t count = 1);

    /**
     * Dynamic State
     */
    std::vector<vk::DynamicState> dynamicStates = { vk::DynamicState::eScissor, vk::DynamicState::eViewport };

    /**
     * Graphics Pipeline State Configuration
     */
    vk::PipelineInputAssemblyStateCreateInfo inputAssemblyState = makeInputAssemblyState();
    vk::PipelineRasterizationStateCreateInfo rasterizationState = makeRasterizationState();
    vk::PipelineMultisampleStateCreateInfo   multisampleState   = makeMultisampleState();
    vk::PipelineDepthStencilStateCreateInfo  depthStencilState  = makeDepthStencilState();
    vk::PipelineViewportStateCreateInfo      viewportState      = makeViewportState();
    vk::PipelineDynamicStateCreateInfo       dynamicState       = makeDynamicState();
    vk::PipelineColorBlendStateCreateInfo    colorBlendState    = makeColorBlendState();
    vk::PipelineVertexInputStateCreateInfo   vertexInputState   = makeVertexInputState();

    GraphicsPS& setCullMode(vk::CullModeFlagBits cullMode);

    GraphicsPS& setSampleCount(vk::SampleCountFlagBits samples);

    GraphicsPS& setTopology(vk::PrimitiveTopology topology);

    GraphicsPS& setWireframeMode(bool value = false);

    /**
     * Freely configure the pipeline state via a lambda function.
     */
    template<std::invocable<GraphicsPS&> Fn>
    GraphicsPS& configure(Fn&& fn)
    {
        std::forward<Fn>(fn)(*this);
        return *this;
    }

private:
    [[nodiscard]] static vk::PipelineInputAssemblyStateCreateInfo makeInputAssemblyState();

    [[nodiscard]] static vk::PipelineRasterizationStateCreateInfo makeRasterizationState();

    [[nodiscard]] static vk::PipelineMultisampleStateCreateInfo makeMultisampleState();

    [[nodiscard]] static vk::PipelineDepthStencilStateCreateInfo makeDepthStencilState();

    [[nodiscard]] static vk::PipelineViewportStateCreateInfo makeViewportState();

    [[nodiscard]] static vk::PipelineDynamicStateCreateInfo makeDynamicState();

    [[nodiscard]] static vk::PipelineColorBlendStateCreateInfo makeColorBlendState();

    [[nodiscard]] static vk::PipelineVertexInputStateCreateInfo makeVertexInputState();
};

class PipelineBase
{
protected:
    // Used internally for pipeline creation
    struct LoadedShader
    {
        ShaderInfo shaderInfo;
        std::vector<char> code;
    };
public:
    explicit PipelineBase(const PipelineCommon& common, PipelineType type, const std::shared_ptr<Device>& device);

    ~PipelineBase();

    [[nodiscard]] const vk::Pipeline& getHandle() const noexcept;

    [[nodiscard]] PipelineType getType() const noexcept;

    [[nodiscard]] const vk::PipelineBindPoint& getBindPoint() const noexcept;

    std::string label;

protected:
    // Convert from pipeline type to Vulkan bind point.
    [[nodiscard]] static vk::PipelineBindPoint toBindPoint(PipelineType type);

    // Read a shader file from disk.
    [[nodiscard]] static std::vector<char> readShaderFile(const std::filesystem::path& filePath);

    // Load a compiled SPIR-V shader from disk.
    [[nodiscard]] static LoadedShader loadShader(const ShaderInfo& shaderInfo);

    std::shared_ptr<Device> mDevice     = nullptr;
    vk::Pipeline            mPipeline   = VK_NULL_HANDLE;
    PipelineType            mType       = PipelineType::Unknown;
    vk::PipelineBindPoint   mBindPoint  = {};
};

class ComputePipeline : public PipelineBase
{
public:
    explicit ComputePipeline(const PipelineCommon& common, const std::shared_ptr<Device>& device)
    : PipelineBase(common, PipelineType::Compute, device)
    {
        const auto computeShaders = common.shaders
            | std::views::filter([](const ShaderInfo& info) {
                return info.stage == vk::ShaderStageFlagBits::eCompute;
            })
            | std::ranges::to<std::vector>();
        if (computeShaders.empty())
        {
            throw std::runtime_error(fmt::format("No compute shader specified for pipeline [{}]", label));
        }

        const auto shader = loadShader(computeShaders[0]);
        const auto moduleInfo = vk::ShaderModuleCreateInfo()
            .setCodeSize(sizeof(char) * shader.code.size())
            .setPCode(reinterpret_cast<const uint32_t*>(shader.code.data()));
        const auto stageInfo = vk::PipelineShaderStageCreateInfo()
            .setStage(shader.shaderInfo.stage)
            .setPName(shader.shaderInfo.entryPoint)
            .setPNext(&moduleInfo);

        constexpr auto flags = vk::PipelineCreateFlags2CreateInfo()
            .setFlags(vk::PipelineCreateFlagBits2::eDescriptorHeapEXT);
        const auto pipelineCreateInfo = vk::ComputePipelineCreateInfo()
            .setLayout(nullptr)
            .setStage(stageInfo)
            .setPNext(&flags);

        try
        {
            mPipeline = mDevice->getHandle().createComputePipeline(nullptr, pipelineCreateInfo).value;
            mDevice->setLabel(mPipeline, fmt::format("{}_Pipeline", label));
        }
        catch (const vk::SystemError& err)
        {
            throw std::runtime_error(fmt::format("Failed to create Pipeline [{}]: {}", label, err.what()));
        }
    }
};

class GraphicsPipeline : public PipelineBase
{
public:
    explicit GraphicsPipeline(GraphicsPS ps, const PipelineCommon& common, const std::shared_ptr<Device>& device)
    : PipelineBase(common, PipelineType::Graphics, device)
    {
        ps.colorBlendState.setAttachments(ps.blendStates);
        ps.vertexInputState.setVertexAttributeDescriptions(ps.vtxInputAttributes);
        ps.vertexInputState.setVertexBindingDescriptions(ps.vtxInputBindings);
        ps.dynamicState.setDynamicStates(ps.dynamicStates);

        std::vector<LoadedShader> shaders;
        shaders.reserve(common.shaders.size());
        for (const auto& shaderInfo : common.shaders)
        {
            shaders.push_back(loadShader(shaderInfo));
        }

        std::vector<vk::PipelineShaderStageCreateInfo> stageInfos(shaders.size());
        std::vector<vk::ShaderModuleCreateInfo>        moduleInfos(shaders.size());
        for (const auto& [i, shader] : std::views::enumerate(shaders))
        {
            moduleInfos[i] = vk::ShaderModuleCreateInfo()
                .setCodeSize(sizeof(char) * shader.code.size())
                .setPCode(reinterpret_cast<const uint32_t*>(shader.code.data()));

            stageInfos[i] = vk::PipelineShaderStageCreateInfo()
                .setStage(shader.shaderInfo.stage)
                .setPName(shader.shaderInfo.entryPoint)
                .setPNext(&moduleInfos[i]);
        }

        const auto renderingInfo = vk::PipelineRenderingCreateInfo()
            .setColorAttachmentFormats(ps.attachmentFormats)
            .setDepthAttachmentFormat(ps.depthFormat)
            .setStencilAttachmentFormat(ps.stencilFormat);

        const auto flags = vk::PipelineCreateFlags2CreateInfo()
            .setFlags(vk::PipelineCreateFlagBits2::eDescriptorHeapEXT)
            .setPNext(&renderingInfo);
        auto graphicsPipelineCreateInfo = vk::GraphicsPipelineCreateInfo()
            .setPInputAssemblyState(&ps.inputAssemblyState)
            .setPRasterizationState(&ps.rasterizationState)
            .setPMultisampleState(&ps.multisampleState)
            .setPDepthStencilState(&ps.depthStencilState)
            .setPViewportState(&ps.viewportState)
            .setPDynamicState(&ps.dynamicState)
            .setPColorBlendState(&ps.colorBlendState)
            .setPVertexInputState(&ps.vertexInputState)
            .setStages(stageInfos)
            .setLayout(nullptr)
            .setPNext(&flags);

        try
        {
            mPipeline = mDevice->getHandle().createGraphicsPipeline(nullptr, graphicsPipelineCreateInfo).value;
            mDevice->setLabel(mPipeline, label);
        }
        catch (const vk::SystemError& err)
        {
            throw std::runtime_error(fmt::format("Failed to create Pipeline [{}]: {}", label, err.what()));
        }
    }
};

#pragma endregion

class CommandList final
{
public:
    CommandList(const vk::CommandBuffer& commandBuffer, const bool isSingleTime)
    : mHandle(commandBuffer)
    , mIsSingleTime(isSingleTime)
    {
    }

    [[nodiscard]] vk::CommandBuffer getHandle() const noexcept
    {
        return mHandle;
    }

    void begin() const
    {
        auto beginInfo = vk::CommandBufferBeginInfo();
        if (mIsSingleTime)
        {
            beginInfo.setFlags(vk::CommandBufferUsageFlagBits::eOneTimeSubmit);
        }
        mHandle.begin(beginInfo);
    }

    void end() const
    {
        mHandle.end();
    }

    void setViewportScissor(const vk::Viewport& viewport, const vk::Rect2D scissor) const
    {
        mHandle.setViewport(0, viewport);
        mHandle.setScissor(0, scissor);
    }

    void beginLabel(const std::string& name) const
    {
        const auto label = vk::DebugUtilsLabelEXT().setPLabelName(name.c_str());
        mHandle.beginDebugUtilsLabelEXT(label);
    }

    void endLabel() const
    {
        mHandle.endDebugUtilsLabelEXT();
    }

    void bindPipeline(PipelineBase* pPipeline)
    {
        mBoundPipeline = pPipeline;
        mHandle.bindPipeline(pPipeline->getBindPoint(), pPipeline->getHandle());
    }

    template <class T>
    void pushData(const T& data)
    {
        const auto pushDataInfo = vk::PushDataInfoEXT()
            .setData(vk::HostAddressRangeConstEXT().setAddress(&data).setSize(sizeof(T)))
            .setOffset(0);
        mHandle.pushDataEXT(pushDataInfo);
    }

    void dispatch(const uint32_t x = 1, const uint32_t y = 1, const uint32_t z = 1) const
    {
        mHandle.dispatch(x, y, z);
    }

    void dispatchIndirect(const Buffer* pBuffer) const
    {
        mHandle.dispatchIndirect(pBuffer->getHandle(), 0);
    }

    void drawMeshTasks(const uint32_t x = 1, const uint32_t y = 1, const uint32_t z = 1) const
    {
        mHandle.drawMeshTasksEXT(x, y, z);
    }

    void blitToSwapchain(Texture* pSrc, const Swapchain* pSwapchain, uint32_t acquiredIndex) const;

private:
    vk::CommandBuffer mHandle;
    bool              mIsSingleTime  = false;
    PipelineBase*     mBoundPipeline = nullptr;
};

class CommandPool final
{
public:
    CommandPool(const std::shared_ptr<Device>& device, const DeviceQueue& queue)
    : mDevice(device)
    {
        const auto poolCreateInfo = vk::CommandPoolCreateInfo()
            .setQueueFamilyIndex(queue.familyIndex)
            .setFlags(vk::CommandPoolCreateFlagBits::eResetCommandBuffer);
        mCommandPool = mDevice->getHandle().createCommandPool(poolCreateInfo);
    }

    ~CommandPool()
    {
        mDevice->getHandle().destroy(mCommandPool);
    }

    CommandList* allocate()
    {
        const auto allocInfo = vk::CommandBufferAllocateInfo()
            .setCommandBufferCount(1)
            .setCommandPool(mCommandPool)
            .setLevel(vk::CommandBufferLevel::ePrimary);

        const auto h = mDevice->getHandle().allocateCommandBuffers(allocInfo);

        auto commandList = std::make_unique<CommandList>(h[0], false);
        mCommandLists.push_back(std::move(commandList));

        auto* pList = mCommandLists[mCommandLists.size() - 1].get();
        return pList;
    }

    void free(const CommandList* pCommandList)
    {
        const auto commandBuffer = pCommandList->getHandle();
        mDevice->getHandle().freeCommandBuffers(mCommandPool, 1, &commandBuffer);
        std::erase_if(mCommandLists, [pCommandList](const auto& list) { return list.get() == pCommandList; });
    }

    void reset() const
    {
        mDevice->getHandle().resetCommandPool(mCommandPool);
    }

    [[nodiscard]] vk::CommandPool getHandle() const noexcept
    {
        return mCommandPool;
    }

private:
    std::shared_ptr<Device>                     mDevice;
    vk::CommandPool                             mCommandPool;
    std::vector<std::unique_ptr<CommandList>>   mCommandLists;
};

class CommandQueue final
{
public:
    CommandQueue(const std::shared_ptr<Device>& device, const DeviceQueue& queue)
    : mDevice(device)
    , mQueue(queue)
    {
        mImmediatePool = std::make_unique<CommandPool>(mDevice, mQueue);
    }

    void waitIdle() const
    {
        mQueue.queue.waitIdle();
    }

    void immediate(const std::function<void(CommandList*)>& fn) const
    {
        auto* pCmdList = mImmediatePool->allocate();

        pCmdList->begin();
        fn(pCmdList);
        pCmdList->end();

        const auto cmdInfo    = vk::CommandBufferSubmitInfo().setCommandBuffer(pCmdList->getHandle());
        const auto submitInfo = vk::SubmitInfo2().setCommandBufferInfos(cmdInfo);
        mQueue.queue.submit2(submitInfo, nullptr);
        waitIdle();

        mImmediatePool->free(pCmdList);
    }

    [[nodiscard]] std::unique_ptr<CommandPool> spawnPool() const
    {
        return std::make_unique<CommandPool>(mDevice, mQueue);
    }

private:
    std::shared_ptr<Device>      mDevice;
    DeviceQueue                  mQueue;
    std::unique_ptr<CommandPool> mImmediatePool;
};

class Barrier
{
public:
    Barrier& addBarrier(const vk::BufferMemoryBarrier2& bufferBarrier)
    {
        mBufferBarriers.push_back(bufferBarrier);
        return *this;
    }

    Barrier& addBarrier(const vk::ImageMemoryBarrier2& imageBarrier)
    {
        mImageBarriers.push_back(imageBarrier);
        return *this;
    }

    void insert(const CommandList* pCommandList) const
    {
        const auto dependencyInfo = vk::DependencyInfo()
            .setImageMemoryBarriers(mImageBarriers)
            .setBufferMemoryBarriers(mBufferBarriers);
        pCommandList->getHandle().pipelineBarrier2(dependencyInfo);
    }

private:
    std::vector<vk::BufferMemoryBarrier2>   mBufferBarriers;
    std::vector<vk::ImageMemoryBarrier2>    mImageBarriers;
};

struct Frame
{
    // Assigned command list
    CommandList*   pCommandList;
    // Current in-flight frame index (frameNumber mod FinF)
    const uint64_t currentFrame;
    // Lifetime frame number since app start
    const uint64_t frameNumber;
    // The swapchain image index for this frame
    const uint32_t acquiredIndex;
};

enum class DescriptorHeapType
{
    Resource,
    Sampler,
};

class DescriptorHeap
{
public:
    explicit DescriptorHeap(
        const std::shared_ptr<Device>& device,
        const std::string&             label,
        const DescriptorHeapType       type = DescriptorHeapType::Resource,
        const uint32_t                 capacity = 1 << 16)
    : type(type)
    , capacity(capacity)
    , label(label)
    , mDevice(device)
    {
        const auto props = mDevice->getPhysicalDevice()
            .getProperties2<
                vk::PhysicalDeviceProperties2,
                vk::PhysicalDeviceDescriptorHeapPropertiesEXT>()
            .get<vk::PhysicalDeviceDescriptorHeapPropertiesEXT>();

        vk::DeviceSize heapAlignment = 0;
        vk::DeviceSize maxHeapSize   = 0;

        if (type == DescriptorHeapType::Resource)
        {
            mStride       = alignUp(props.imageDescriptorSize, props.imageDescriptorAlignment);
            mReservedSize = props.minResourceHeapReservedRange;
            heapAlignment = props.resourceHeapAlignment;
            maxHeapSize   = props.maxResourceHeapSize;
        }
        else
        {
            mStride       = alignUp(props.samplerDescriptorSize, props.samplerDescriptorAlignment);
            mReservedSize = props.minSamplerHeapReservedRange;
            heapAlignment = props.samplerHeapAlignment;
            maxHeapSize   = props.maxSamplerHeapSize;
        }

        const vk::DeviceSize size = alignUp(mStride * capacity + mReservedSize, heapAlignment);
        mBuffer = Buffer::Builder()
            .setSize(size)
            .setType(BufferType::DescriptorHeap)
            .setLabel(label)
            .create(mDevice);

        mMapped = static_cast<std::byte*>(mBuffer->getMapped());
    }

    ~DescriptorHeap() = default;

    void writeSampledImage(const uint32_t index, const Texture* pTexture) const
    {
        writeImage(index, pTexture, vk::DescriptorType::eSampledImage);
    }

    void writeStorageImage(const uint32_t index, const Texture* pTexture) const
    {
        writeImage(index, pTexture, vk::DescriptorType::eStorageImage);
    }

    void writeSampler(const uint32_t index, const vk::SamplerCreateInfo& info) const
    {
        assert(type == DescriptorHeapType::Sampler);
        mDevice->getHandle().writeSamplerDescriptorsEXT(info, slot(index));
    }

    void bind(const CommandList* pCommandList) const
    {
        vk::BindHeapInfoEXT info{};
        info.heapRange           = vk::DeviceAddressRangeEXT{mBuffer->getAddress(), mBuffer->size};
        info.reservedRangeOffset = mBuffer->size - mReservedSize;
        info.reservedRangeSize   = mReservedSize;

        if (type == DescriptorHeapType::Resource)
        {
            pCommandList->getHandle().bindResourceHeapEXT(info);
        }
        else
        {
            pCommandList->getHandle().bindSamplerHeapEXT(info);
        }
    }

    const DescriptorHeapType type;
    const uint32_t           capacity;
    const std::string        label;

private:
    void writeImage(const uint32_t index, const Texture* pTexture, const vk::DescriptorType dType) const
    {
        assert(type == DescriptorHeapType::Resource);
        const auto image = vk::ImageDescriptorInfoEXT()
            .setLayout(vk::ImageLayout::eGeneral)
            .setPView(&pTexture->getImageViewCreateInfo());
        const auto data = vk::ResourceDescriptorDataEXT()
            .setPImage(&image);
        const auto info = vk::ResourceDescriptorInfoEXT()
            .setType(dType)
            .setData(data);

        const auto dst = slot(index);
        mDevice->getHandle().writeResourceDescriptorsEXT(info, dst);
    }

    [[nodiscard]] vk::HostAddressRangeEXT slot(const uint32_t index) const noexcept
    {
        return vk::HostAddressRangeEXT()
            .setAddress(mMapped + index * mStride)
            .setSize(mStride);
    }

    std::shared_ptr<Device> mDevice;

    std::unique_ptr<Buffer> mBuffer;
    std::byte*              mMapped = nullptr;
    vk::DeviceSize          mStride = 0;
    vk::DeviceSize          mReservedSize = 0;

};

struct Rendering
{
    // Set the command list to be used across the rendering interval.
    Rendering& usingCommandList(CommandList* pCommandList)
    {
        mCommandList = pCommandList;
        return *this;
    }

    //Set the debug marker region label for the render pass body.
    Rendering& setLabel(const std::string& value);

    // Set the render area with an extent, the offset will be set to {0, 0}.
    Rendering& setRenderArea(const vk::Extent2D& extent);

    // Set the render area with a rect.
    Rendering& setRenderArea(const vk::Rect2D& rect);

    // Add a color or depth attachment to the render pass. The last attachment added is used for render area, unless specifed.
    Rendering& addAttachment(
        Texture*                             pImage,
        vk::AttachmentLoadOp                 loadOp     = vk::AttachmentLoadOp::eClear,
        vk::AttachmentStoreOp                storeOp    = vk::AttachmentStoreOp::eStore,
        const std::optional<vk::ClearValue>& clearValue = std::nullopt,
        Texture*                             pResolve   = nullptr
    );

    // Get al the barriers required by the attachments for this render pass.
    [[nodiscard]] Barrier getBarriers();

    // Execute the render pass with the configured attachments with its body defined in a lambda.
    template<std::invocable<CommandList*> Fn>
    void execute(Fn&& lambda, const bool bSetViewport = true, const bool bInsertBarriers = true)
    {
        mCommandList->beginLabel(mLabel);

        if (bSetViewport)
        {
            setViewportScissor(mCommandList);
        }
        if (bInsertBarriers)
        {
            insertBarriers(mCommandList);
        }

        const auto renderingInfo = vk::RenderingInfo()
            .setRenderArea(mRenderArea)
            .setLayerCount(1)
            .setColorAttachments(mAttachmentInfos)
            .setPDepthAttachment(mDepthAttachmentInfo.has_value() ? &(*mDepthAttachmentInfo) : nullptr);
        mCommandList->getHandle().beginRendering(&renderingInfo);
        lambda(mCommandList);
        mCommandList->getHandle().endRendering();

        mCommandList->endLabel();
    }

private:
    // Insert all the barriers required for the attachments.
    Rendering& insertBarriers(const CommandList* pCommandList);

    // Use the specified render area for viewport and scissor configuration.
    Rendering& setViewportScissor(const CommandList* pCommandList);

    CommandList*                               mCommandList = nullptr;

    bool                                       mAutoRenderArea = true;
    vk::Rect2D                                 mRenderArea;

    std::string                                mLabel;
    std::vector<Texture*>                      mImages;
    std::vector<Texture*>                      mResolveImages;
    std::vector<vk::RenderingAttachmentInfo>   mAttachmentInfos;
    std::optional<vk::RenderingAttachmentInfo> mDepthAttachmentInfo = std::nullopt;
};

class VulkanRHI
{
public:
    explicit VulkanRHI(SDL_Window* pWindow)
    {
        int32_t w, h;
        SDL_GetWindowSizeInPixels(pWindow, &w, &h);
        mRenderResolution.setWidth(w).setHeight(h);

        const vk::detail::DynamicLoader dynamicLoader;
        const auto vkGetInstanceProcAddr = dynamicLoader.getProcAddress<PFN_vkGetInstanceProcAddr>("vkGetInstanceProcAddr");
        VULKAN_HPP_DEFAULT_DISPATCHER.init(vkGetInstanceProcAddr);

        mInstance = std::make_unique<Instance>();
        VULKAN_HPP_DEFAULT_DISPATCHER.init(mInstance->getHandle());

        mDevice = std::make_shared<Device>(mInstance.get());
        VULKAN_HPP_DEFAULT_DISPATCHER.init(mDevice->getHandle());

        mSwapchain = std::make_unique<Swapchain>(pWindow, mInstance.get(), mRenderResolution, mDevice);

        /* Frame sync objects */ {
            constexpr auto semaphoreCreateInfo = vk::SemaphoreCreateInfo();
            constexpr auto fenceCreateInfo     = vk::FenceCreateInfo().setFlags(vk::FenceCreateFlagBits::eSignaled);
            for (uint32_t i = 0; i < gFramesInFlight; i++)
            {
                mFrameInFlight[i]     = mDevice->getHandle().createFence(fenceCreateInfo);
                mImageReady[i]        = mDevice->getHandle().createSemaphore(semaphoreCreateInfo);
                mRenderingFinished[i] = mDevice->getHandle().createSemaphore(semaphoreCreateInfo);
            }
        }

        mCommandQueue = std::make_unique<CommandQueue>(mDevice, mDevice->getGraphicsQueue());
        mCommandPool  = mCommandQueue->spawnPool();
        for (uint32_t i = 0; i < gFramesInFlight; i++)
        {
            mCommandLists[i] = mCommandPool->allocate();
        }
    }

    ~VulkanRHI()
    {
        mDevice->getHandle().waitIdle();
        for (uint32_t i = 0; i < gFramesInFlight; i++)
        {
            mDevice->getHandle().destroy(mFrameInFlight[i]);
            mDevice->getHandle().destroy(mImageReady[i]);
            mDevice->getHandle().destroy(mRenderingFinished[i]);
        }
    }

    [[nodiscard]] Frame beginFrame() const
    {
        if (const auto result = mDevice->getHandle().waitForFences(mFrameInFlight[mCurrentFrame], true, std::numeric_limits<uint64_t>::max());
            result != vk::Result::eSuccess)
        {
            throw std::runtime_error(fmt::format("Fence wait timed out: {}", vk::to_string(result)));
        }
        mDevice->getHandle().resetFences(mFrameInFlight[mCurrentFrame]);

        const auto acquireNextImageInfo = vk::AcquireNextImageInfoKHR()
            .setSwapchain(mSwapchain->getHandle())
            .setTimeout(std::numeric_limits<uint64_t>::max())
            .setSemaphore(mImageReady[mCurrentFrame])
            .setDeviceMask(1);
        const uint32_t acquiredIndex = mDevice->getHandle().acquireNextImage2KHR(acquireNextImageInfo).value;

        auto* pCommandList = mCommandLists[mCurrentFrame];
        pCommandList->begin();

        return {
            .pCommandList  = pCommandList,
            .currentFrame  = mCurrentFrame,
            .frameNumber   = mFrameCounter,
            .acquiredIndex = acquiredIndex,
        };
    }

    void endFrameAndPresent(const Frame& frame)
    {
        frame.pCommandList->end();

        const auto cmdInfo    = vk::CommandBufferSubmitInfo().setCommandBuffer(frame.pCommandList->getHandle());
        const auto waitInfo   = vk::SemaphoreSubmitInfo().setSemaphore(mImageReady[mCurrentFrame]).setStageMask(vk::PipelineStageFlagBits2::eAllCommands);
        const auto signalInfo = vk::SemaphoreSubmitInfo().setSemaphore(mRenderingFinished[mCurrentFrame]).setStageMask(vk::PipelineStageFlagBits2::eAllCommands);

        const auto submitInfo = vk::SubmitInfo2()
            .setCommandBufferInfos(cmdInfo)
            .setWaitSemaphoreInfos(waitInfo)
            .setSignalSemaphoreInfos(signalInfo);

        mDevice->getGraphicsQueue().queue.submit2(submitInfo, mFrameInFlight[mCurrentFrame]);

        const auto swapchain = mSwapchain->getHandle();
        const auto presentInfo = vk::PresentInfoKHR()
            .setWaitSemaphores(mRenderingFinished[mCurrentFrame])
            .setSwapchains(swapchain)
            .setImageIndices(frame.acquiredIndex);

        if (const auto result = mDevice->getGraphicsQueue().queue.presentKHR(presentInfo);
            result != vk::Result::eSuccess)
        {
            throw std::runtime_error(fmt::format("Present failed: {}", vk::to_string(result)));
        }

        mFrameCounter++;
        mCurrentFrame = (mCurrentFrame + 1) % gFramesInFlight;
    }

    [[nodiscard]] const Instance* getInstance() const noexcept
    {
        return mInstance.get();
    }

    [[nodiscard]] const std::shared_ptr<Device>& getDevice() const noexcept
    {
        return mDevice;
    }

    [[nodiscard]] const Swapchain* getSwapchain() const noexcept
    {
        return mSwapchain.get();
    }

    [[nodiscard]] const CommandQueue* getCommandQueue() const noexcept
    {
        return mCommandQueue.get();
    }

    [[nodiscard]] vk::Extent2D getRenderResolution() const noexcept
    {
        return mRenderResolution;
    }

private:
    vk::Extent2D                    mRenderResolution;
    std::unique_ptr<Instance>       mInstance;
    std::shared_ptr<Device>         mDevice;
    std::unique_ptr<Swapchain>      mSwapchain;

    std::unique_ptr<CommandQueue>   mCommandQueue;
    std::unique_ptr<CommandPool>    mCommandPool;
    PerFrameArray<CommandList*>     mCommandLists {};

    uint64_t                        mCurrentFrame = 0;
    uint64_t                        mFrameCounter = 0;
    PerFrameArray<vk::Fence>        mFrameInFlight;
    PerFrameArray<vk::Semaphore>    mImageReady;
    PerFrameArray<vk::Semaphore>    mRenderingFinished;
};
