#pragma once

#include <array>
#include <glm/glm.hpp>
#include <glm/gtx/euler_angles.hpp>

#include "Hair/HairBuffers.hpp"
#include "Hair/HairGeometry.hpp"
#include "Hair/HairRendererParams.hpp"
#include "VulkanRHI/VulkanRHI.hpp"

struct GpuScreenSpaceTriangle
{
    glm::vec3 v0, v1, v2;
    glm::vec3 w0, w1, w2;
    glm::vec3 tangent;
    glm::vec3 color;
    uint32_t  primitiveId;
};

class TwoPassHybrid
{
public:
    explicit TwoPassHybrid(HairRendererParams* pParams, VulkanRHI* pRHI)
    : mVulkanRHI(pRHI)
    , mParams(pParams)
    {
        /* Mesh Pipeline */
        {
            const auto graphicsPS = GraphicsPS()
                .addDefaultAttachmentState(1)
                .addAttachmentFormat(vk::Format::eR32G32B32A32Sfloat)
                .addAttachmentFormat(vk::Format::eD32Sfloat)
                .setCullMode(vk::CullModeFlagBits::eNone);
            const auto pipelineInfo = PipelineCommon()
                .setLabel("TwoPass_Mesh")
                .addShader("TwoPass_Mesh.task.spv")
                .addShader("TwoPass_Mesh.mesh.spv")
                .addShader("TwoPass_Mesh.frag.spv");

            mMeshPipeline = std::make_unique<GraphicsPipeline>(graphicsPS, pipelineInfo, mVulkanRHI->getDevice());
        }

        /* Indirect Prepare Pipeline */
        {
            const auto pipelineInfo = PipelineCommon()
                .setLabel("TwoPass_PrepareIndirect")
                .addShader("TwoPass_PrepareIndirect.comp.spv");
            // mPrepareIndirectPipeline = std::make_unique<ComputePipeline>(pipelineInfo, mVulkanRHI->getDevice());
        }

        /* Software Rasterizer Pipeline */
        {
            const auto pipelineInfo = PipelineCommon()
                .setLabel("TwoPass_Rasterizer")
                .addShader("TwoPass_Rasterizer.comp.spv");
            // mComputePipeline = std::make_unique<ComputePipeline>(pipelineInfo, mVulkanRHI->getDevice());
        }

        mDescriptorHeap = std::make_unique<DescriptorHeap>(mVulkanRHI->getDevice(), "TwoPass_DescriptorHeap");

        // Worst case buffer size
        const uint64_t triDataSize = mParams->mHairGeometry->getVertexCount() * 2 * sizeof(GpuScreenSpaceTriangle);

        for (uint32_t i = 0; i < gFramesInFlight; i++)
        {
            mRenderTarget[i] = Texture::Builder()
                .setFormat(vk::Format::eR32G32B32A32Sfloat)
                .setSize(mVulkanRHI->getRenderResolution())
                .setLabel(fmt::format("TwoPass_Target_{}", i))
                .create(mVulkanRHI->getDevice());

            mDepthBuffers[i] = Texture::Builder()
                .setFormat(vk::Format::eD32Sfloat)
                .setSize(mVulkanRHI->getRenderResolution())
                .setLabel(fmt::format("TwoPass_Depth_{}", i))
                .create(mVulkanRHI->getDevice());

            mTriangleDataBuffers[i] = Buffer::Builder()
                .setSize(triDataSize)
                .setType(BufferType::Storage)
                .setLabel(fmt::format("TwoPass_TriData_{}", i))
                .create(mVulkanRHI->getDevice());

            mTriangleCountBuffers[i] = Buffer::Builder()
                .setSize(sizeof(uint32_t))
                .setType(BufferType::Storage)
                .setLabel(fmt::format("TwoPass_TriCount_{}", i))
                .create(mVulkanRHI->getDevice());

            mIndirectArgsBuffers[i] = Buffer::Builder()
                .setSize(sizeof(uint32_t) * 3)
                .setType(BufferType::Indirect)
                .setLabel(fmt::format("TwoPass_IndirectArgsBuf_{}", i))
                .create(mVulkanRHI->getDevice());

            mDescriptorHeap->writeStorageImage((i * 2) + 0, mRenderTarget[i].get());
            mDescriptorHeap->writeStorageImage((i * 2) + 1, mDepthBuffers[i].get());
        }
    }

    void render(const Frame& frame, const HairBuffers& buffers) const
    {
        mDescriptorHeap->bind(frame.pCommandList);

        clearCounterBuffer(frame);
        dispatchMesh(frame, buffers);
        // dispatchCompute(frame, buffers);

        frame.pCommandList->blitToSwapchain(mRenderTarget[frame.currentFrame].get(), mVulkanRHI->getSwapchain(), frame.acquiredIndex);
    }

private:
    void clearCounterBuffer(const Frame& frame) const
    {
        using enum BufferUsage;
        const auto* pBuffer = mTriangleCountBuffers[frame.currentFrame].get();
        Barrier()
            .addBarrier(pBuffer->getBarrier(All, TransferDst))
            .insert(frame.pCommandList);
        frame.pCommandList->getHandle().fillBuffer(pBuffer->getHandle(), 0, sizeof(uint32_t), 0);
    }

    void dispatchMesh(const Frame& frame, const HairBuffers& buffers) const
    {
        using enum BufferUsage;

        const auto* pTriData = mTriangleDataBuffers[frame.currentFrame].get();
        const auto* pTriCount = mTriangleCountBuffers[frame.currentFrame].get();

        Barrier()
            .addBarrier(pTriData->getBarrier(TransferDst, Compute_Write))
            .addBarrier(pTriCount->getBarrier(TransferDst, Compute_Write))
            .insert(frame.pCommandList);

        Rendering()
            .setLabel("Hair-TwoPass-MeshStage")
            .usingCommandList(frame.pCommandList)
            .addAttachment(mRenderTarget[frame.currentFrame].get())
            .addAttachment(mDepthBuffers[frame.currentFrame].get())
            .execute([&](CommandList* pCmd) -> void
            {
                const auto [w, h] = mVulkanRHI->getRenderResolution();
                const struct
                {
                    glm::mat4   model;
                    HairBuffers buffers;
                    uint64_t    pTriData;
                    uint64_t    pTriCount;
                    uint32_t    strandCount;
                    uint32_t    colorMode;
                    glm::vec2   viewport;
                    float       threshold;
                } pushData = {
                    .model       = glm::yawPitchRoll(0.0f, glm::radians(-90.0f), 0.0f),
                    .buffers     = buffers,
                    .pTriData    = pTriData->getAddress(),
                    .pTriCount   = pTriCount->getAddress(),
                    .strandCount = static_cast<uint32_t>(mParams->mStrandCount),
                    .colorMode   = std::to_underlying(mParams->mColorMode),
                    .viewport    = glm::vec2(w, h),
                    .threshold   = mParams->mTriThreshold,
                };
                pCmd->bindPipeline(mMeshPipeline.get());
                pCmd->pushData(pushData);
                pCmd->drawMeshTasks(mParams->mHairGeometry->taskGroupSizeX);
            });
    }

    void dispatchCompute(const Frame& frame, const HairBuffers& buffers) const
    {
        using enum BufferUsage;
        auto* pCmd = frame.pCommandList;

        const auto* pTriData  = mTriangleDataBuffers[frame.currentFrame].get();
        const auto* pTriCount = mTriangleCountBuffers[frame.currentFrame].get();
        const auto* pArgs     = mIndirectArgsBuffers[frame.currentFrame].get();

        /* Prepare Indirect Dispatch */
        {
            const uint32_t maxSmallTriangles = pTriData->size / sizeof(GpuScreenSpaceTriangle);
            const uint32_t maxGroups = (maxSmallTriangles + 64 - 1) / 64; // Where group size is 64

            Barrier()
                .addBarrier(pTriCount->getBarrier(All, Compute_Read))
                .addBarrier(pArgs->getBarrier(All, Compute_Write))
                .insert(pCmd);

            const struct
            {
                uint64_t pTriCount;
                uint64_t pIndirectArgs;
                uint32_t maxGroups;
            } pushData = {
                .pTriCount     = pTriCount->getAddress(),
                .pIndirectArgs = pArgs->getAddress(),
                .maxGroups     = maxGroups,
            };

            pCmd->bindPipeline(mPrepareIndirectPipeline.get());
            pCmd->pushData(pushData);
            pCmd->dispatch();
        }

        /* Rasterizer */
        {
            Barrier()
                .addBarrier(pTriData->getBarrier(Compute_Write, Compute_Read))
                .addBarrier(pArgs->getBarrier(Compute_Write, DrawIndirect))
                .addBarrier(mRenderTarget[frame.currentFrame]->getBarrier(ImageUsage::StorageImage))
                .addBarrier(mDepthBuffers[frame.currentFrame]->getBarrier(ImageUsage::StorageImage))
                .insert(pCmd);

            const auto [w, h] = mVulkanRHI->getRenderResolution();
            const struct
            {
                HairBuffers buffers;
                uint64_t    pTriCount;
                uint64_t    pTriData;
                glm::vec2   viewportSize;
                uint32_t    renderTarget;
                uint32_t    depthBuffer;
            } pushData = {
                .buffers      = buffers,
                .pTriCount    = pTriCount->getAddress(),
                .pTriData     = pTriData->getAddress(),
                .viewportSize = glm::vec2(w, h),
                .renderTarget = static_cast<uint32_t>(frame.currentFrame * 2) + 0,
                .depthBuffer  = static_cast<uint32_t>(frame.currentFrame * 2) + 1,
            };

            pCmd->bindPipeline(mComputePipeline.get());
            pCmd->pushData(pushData);
            pCmd->dispatchIndirect(pArgs);
        }
    }

    VulkanRHI*                              mVulkanRHI;
    HairRendererParams*                     mParams;

    std::unique_ptr<GraphicsPipeline>       mMeshPipeline;
    std::unique_ptr<ComputePipeline>        mPrepareIndirectPipeline;
    std::unique_ptr<ComputePipeline>        mComputePipeline;

    std::unique_ptr<DescriptorHeap>         mDescriptorHeap;

    PerFrameArray<std::unique_ptr<Texture>> mRenderTarget;
    PerFrameArray<std::unique_ptr<Texture>> mDepthBuffers;

    PerFrameArray<std::unique_ptr<Buffer>>  mIndirectArgsBuffers;

    PerFrameArray<std::unique_ptr<Buffer>>  mTriangleDataBuffers;
    PerFrameArray<std::unique_ptr<Buffer>>  mTriangleCountBuffers;
};
