#pragma once

#include <array>
#include <glm/glm.hpp>
#include <glm/gtx/euler_angles.hpp>

#include "Hair/HairBuffers.hpp"
#include "Hair/HairGeometry.hpp"
#include "Hair/HairRendererParams.hpp"
#include "VulkanRHI/VulkanRHI.hpp"

class CombinedHybridV2
{
public:
    explicit CombinedHybridV2(HairRendererParams* pParams, VulkanRHI* pRHI)
    : mVulkanRHI(pRHI)
    , mParams(pParams)
    {
        /* Mesh Pipeline */
        {
            const auto graphicsPS = GraphicsPS()
                .addAttachmentFormat(vk::Format::eD32Sfloat)
                .setCullMode(vk::CullModeFlagBits::eNone);
            const auto pipelineInfo = PipelineCommon()
                .setLabel("OnePassV2")
                .addShader("OnePassV2.task.spv")
                .addShader("OnePassV2.mesh.spv")
                .addShader("OnePassV2.frag.spv");

            mMeshPipeline = std::make_unique<GraphicsPipeline>(graphicsPS, pipelineInfo, mVulkanRHI->getDevice());
        }

        /* Resolve Pipeline */
        {
            const auto pipelineInfo = PipelineCommon()
                .setLabel("OnePassV2_Resolve")
                .addShader("OnePassV2.comp.spv");
            mResolvePipeline = std::make_unique<ComputePipeline>(pipelineInfo, mVulkanRHI->getDevice());
        }

        mDescriptorHeap = std::make_unique<DescriptorHeap>(mVulkanRHI->getDevice(), "OnePassV2_DescriptorHeap");

        for (uint32_t i = 0; i < gFramesInFlight; i++)
        {
            mRenderTarget[i] = Texture::Builder()
                .setFormat(vk::Format::eR16G16B16A16Sfloat)
                .setSize(mVulkanRHI->getRenderResolution())
                .setLabel(fmt::format("OnePassV2_Target_{}", i))
                .create(mVulkanRHI->getDevice());

            mDepthColor[i] = Texture::Builder()
                .setFormat(vk::Format::eR64Uint)
                .setSize(mVulkanRHI->getRenderResolution())
                .setLabel(fmt::format("OnePassV2_DepthColor_{}", i))
                .create(mVulkanRHI->getDevice());

            mDepthBuffers[i] = Texture::Builder()
                .setFormat(vk::Format::eD32Sfloat)
                .setSize(mVulkanRHI->getRenderResolution())
                .setLabel(fmt::format("OnePassV2_Depth_{}", i))
                .create(mVulkanRHI->getDevice());

            mDescriptorHeap->writeStorageImage((i * 2) + 0, mRenderTarget[i].get());
            mDescriptorHeap->writeStorageImage((i * 2) + 1, mDepthColor[i].get());
        }
    }

    void render(const Frame& frame, const HairBuffers& buffers) const
    {
        mDescriptorHeap->bind(frame.pCommandList);

        clearDepthColor(frame);
        dispatchMesh(frame, buffers);
        dispatchResolve(frame, buffers);

        frame.pCommandList->blitToSwapchain(mRenderTarget[frame.currentFrame].get(), mVulkanRHI->getSwapchain(), frame.acquiredIndex);
    }

private:
    struct PushData
    {
        HairBuffers buffers;
        glm::uvec2  renderTarget;
        glm::uvec2  depthColor;
        glm::mat4   model;
        glm::vec2   viewportSize;
        uint32_t    strandCount;
        float       lineThreshold;
    };

    PushData getPushData(const Frame& frame, const HairBuffers& buffers) const
    {
        const auto [w, h] = mVulkanRHI->getRenderResolution();
        return {
            .buffers       = buffers,
            .renderTarget  = { static_cast<uint32_t>(frame.currentFrame * 2) + 0, 0 },
            .depthColor    = { static_cast<uint32_t>(frame.currentFrame * 2) + 1, 0 },
            .model         = glm::yawPitchRoll(0.0f, glm::radians(-90.0f), 0.0f),
            .viewportSize  = glm::vec2(w, h),
            .strandCount   = static_cast<uint32_t>(mParams->mStrandCount),
            .lineThreshold = mParams->mLineThreshold,
        };
    }

    void clearDepthColor(const Frame& frame) const
    {
        auto* pImage = mDepthColor[frame.currentFrame].get();

        Barrier()
            .addBarrier(pImage->getBarrier(ImageUsage::Clear))
            .insert(frame.pCommandList);

        frame.pCommandList->getHandle().clearColorImage(
            pImage->getHandle(),
            vk::ImageLayout::eGeneral,
            vk::ClearColorValue().setUint32({ 0xFFFFFFFFu, 0xFFFFFFFFu, 0u, 0u }),
            pImage->getSubresourceRange());
    }

    void dispatchMesh(const Frame& frame, const HairBuffers& buffers) const
    {
        Barrier()
            .addBarrier(mDepthColor[frame.currentFrame]->getBarrier(ImageUsage::StorageImage))
            .insert(frame.pCommandList);

        Rendering()
            .setLabel("Hair-OnePassV2-MeshStage")
            .usingCommandList(frame.pCommandList)
            .addAttachment(mDepthBuffers[frame.currentFrame].get())
            .execute([&](CommandList* pCmd) -> void
            {
                const auto pushData = getPushData(frame, buffers);
                pCmd->bindPipeline(mMeshPipeline.get());
                pCmd->pushData(pushData);
                pCmd->drawMeshTasks(pushData.strandCount);
            });
    }

    void dispatchResolve(const Frame& frame, const HairBuffers& buffers) const
    {
        auto* pCmd = frame.pCommandList;

        Barrier()
            .addBarrier(mDepthColor[frame.currentFrame]->getBarrier(ImageUsage::StorageImage))
            .addBarrier(mRenderTarget[frame.currentFrame]->getBarrier(ImageUsage::StorageImage))
            .insert(pCmd);

        const auto [w, h] = mVulkanRHI->getRenderResolution();
        const auto pushData = getPushData(frame, buffers);

        pCmd->bindPipeline(mResolvePipeline.get());
        pCmd->pushData(pushData);
        pCmd->dispatch((w + 8 - 1) / 8, (h + 8 - 1) / 8);
    }

    VulkanRHI*                              mVulkanRHI;
    HairRendererParams*                     mParams;

    std::unique_ptr<GraphicsPipeline>       mMeshPipeline;
    std::unique_ptr<ComputePipeline>        mResolvePipeline;

    std::unique_ptr<DescriptorHeap>         mDescriptorHeap;

    PerFrameArray<std::unique_ptr<Texture>> mRenderTarget;
    PerFrameArray<std::unique_ptr<Texture>> mDepthColor;
    PerFrameArray<std::unique_ptr<Texture>> mDepthBuffers;
};
