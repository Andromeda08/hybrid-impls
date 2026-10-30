#pragma once

#include <array>
#include <glm/glm.hpp>
#include <glm/gtx/euler_angles.hpp>

#include "Hair/HairBuffers.hpp"
#include "Hair/HairGeometry.hpp"
#include "Hair/HairRendererParams.hpp"
#include "VulkanRHI/VulkanRHI.hpp"

class CombinedHybrid
{
public:
    explicit CombinedHybrid(HairRendererParams* pParams, VulkanRHI* pRHI)
    : mVulkanRHI(pRHI)
    , mParams(pParams)
    {
        /* Mesh Pipeline */
        {
            const auto graphicsPS = GraphicsPS()
                // .addDefaultAttachmentState(1)
                // .addAttachmentFormat(vk::Format::eR16G16B16A16Sfloat)
                // .addAttachmentFormat(vk::Format::eD32Sfloat)
                .setCullMode(vk::CullModeFlagBits::eNone);
            const auto pipelineInfo = PipelineCommon()
                .setLabel("OnePass")
                .addShader("OnePass.task.spv")
                .addShader("OnePass.mesh.spv")
                .addShader("OnePass.frag.spv");

            mMeshPipeline = std::make_unique<GraphicsPipeline>(graphicsPS, pipelineInfo, mVulkanRHI->getDevice());
        }

        mDescriptorHeap = std::make_unique<DescriptorHeap>(mVulkanRHI->getDevice(), "TwoPass_DescriptorHeap");

        for (uint32_t i = 0; i < gFramesInFlight; i++)
        {
            mRenderTarget[i] = Texture::Builder()
                .setFormat(vk::Format::eR16G16B16A16Sfloat)
                .setSize(mVulkanRHI->getRenderResolution())
                .setLabel(fmt::format("TwoPass_Target_{}", i))
                .create(mVulkanRHI->getDevice());

            mDepthBuffers[i] = Texture::Builder()
                .setFormat(vk::Format::eD32Sfloat)
                .setSize(mVulkanRHI->getRenderResolution())
                .setLabel(fmt::format("TwoPass_Depth_{}", i))
                .create(mVulkanRHI->getDevice());

            mSoftwareDepthBuffers[i] = Texture::Builder()
                .setFormat(vk::Format::eR32Uint)
                .setSize(mVulkanRHI->getRenderResolution())
                .setLabel(fmt::format("TwoPass_SoftwareDepth_{}", i))
                .create(mVulkanRHI->getDevice());

            mDescriptorHeap->writeStorageImage((i * 2) + 0, mRenderTarget[i].get());
            mDescriptorHeap->writeStorageImage((i * 2) + 1, mSoftwareDepthBuffers[i].get());
        }
    }

    void render(const Frame& frame, const HairBuffers& buffers) const
    {
        mDescriptorHeap->bind(frame.pCommandList);

        dispatchMesh(frame, buffers);

        frame.pCommandList->blitToSwapchain(mRenderTarget[frame.currentFrame].get(), mVulkanRHI->getSwapchain(), frame.acquiredIndex);
    }

private:
    void clearTargets(const Frame& frame) const
    {
        Barrier()
           .addBarrier(mRenderTarget[frame.currentFrame]->getBarrier(ImageUsage::Clear))
           .addBarrier(mSoftwareDepthBuffers[frame.currentFrame]->getBarrier(ImageUsage::Clear))
           .insert(frame.pCommandList);

        frame.pCommandList->getHandle().clearColorImage(
            mRenderTarget[frame.currentFrame]->getHandle(),
            vk::ImageLayout::eGeneral,
            vk::ClearColorValue().setFloat32({ 0.0f, 0.0f, 0.0f, 1.0f }),
            mRenderTarget[frame.currentFrame]->getSubresourceRange());

        frame.pCommandList->getHandle().clearColorImage(
            mSoftwareDepthBuffers[frame.currentFrame]->getHandle(),
            vk::ImageLayout::eGeneral,
            vk::ClearColorValue().setUint32({ 0xFFFFFFFFu, 0u, 0u, 0u }),
            mSoftwareDepthBuffers[frame.currentFrame]->getSubresourceRange());
    }

    void dispatchMesh(const Frame& frame, const HairBuffers& buffers) const
    {
        using enum BufferUsage;

        clearTargets(frame);

        Barrier()
            .addBarrier(mRenderTarget[frame.currentFrame]->getBarrier(ImageUsage::StorageImage))
            .addBarrier(mSoftwareDepthBuffers[frame.currentFrame]->getBarrier(ImageUsage::StorageImage))
            .insert(frame.pCommandList);

        Rendering()
            .setLabel("Hair-TwoPass-MeshStage")
            .usingCommandList(frame.pCommandList)
            // .addAttachment(mRenderTarget[frame.currentFrame].get())
            // .addAttachment(mDepthBuffers[frame.currentFrame].get())
            .setRenderArea(mVulkanRHI->getRenderResolution())
            .execute([&](CommandList* pCmd) -> void
            {
                const auto [w, h] = mVulkanRHI->getRenderResolution();
                const struct
                {
                    HairBuffers buffers;
                    glm::uvec2  renderTarget;
                    glm::uvec2  depthBuffer;
                    glm::mat4   model;
                    glm::vec2   viewportSize;
                    uint32_t    strandCount;
                    float       lineThreshold;
                } pushData = {
                    .buffers       = buffers,
                    .renderTarget  = { static_cast<uint32_t>(frame.currentFrame * 2) + 0, 0 },
                    .depthBuffer   = { static_cast<uint32_t>(frame.currentFrame * 2) + 1, 0 },
                    .model         = glm::yawPitchRoll(0.0f, glm::radians(-90.0f), 0.0f),
                    .viewportSize  = glm::vec2(w, h),
                    .strandCount   = static_cast<uint32_t>(mParams->mStrandCount),
                    .lineThreshold = mParams->mLineThreshold,
                };
                pCmd->bindPipeline(mMeshPipeline.get());
                pCmd->pushData(pushData);
                pCmd->drawMeshTasks(mParams->mHairGeometry->taskGroupSizeX);
            });
    }

    VulkanRHI*                              mVulkanRHI;
    HairRendererParams*                     mParams;

    std::unique_ptr<GraphicsPipeline>       mMeshPipeline;

    std::unique_ptr<DescriptorHeap>         mDescriptorHeap;

    PerFrameArray<std::unique_ptr<Texture>> mRenderTarget;
    PerFrameArray<std::unique_ptr<Texture>> mDepthBuffers;
    PerFrameArray<std::unique_ptr<Texture>> mSoftwareDepthBuffers;
};
