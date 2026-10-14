#include "BaselinePass.hpp"

#include <glm/glm.hpp>
#include <glm/gtx/euler_angles.hpp>
#include "Hair/HairGeometry.hpp"

BaselinePass::BaselinePass(HairRendererParams* pParams, VulkanRHI* pRHI)
: mVulkanRHI(pRHI)
, mParams(pParams)
{
    assert(mParams && mVulkanRHI);
    const auto graphicsPS = GraphicsPS()
        .addDefaultAttachmentState(1)
        .addAttachmentFormat(vk::Format::eB8G8R8A8Unorm)
        .addAttachmentFormat(vk::Format::eD32Sfloat)
        .setCullMode(vk::CullModeFlagBits::eNone);
    const auto pipelineInfo = PipelineCommon()
        .setLabel("BaselinePass")
        .addShader("Baseline.task.spv")
        .addShader("Baseline.mesh.spv")
        .addShader("Baseline.frag.spv");

    mPipeline = std::make_unique<GraphicsPipeline>(graphicsPS, pipelineInfo, mVulkanRHI->getDevice());
}

void BaselinePass::render(const Frame& frame, const HairBuffers& buffers, Texture* pDepthBuffer) const
{
    auto* pImage = mVulkanRHI->getSwapchain()->getTexture(frame.acquiredIndex);
    Rendering()
        .setLabel("Hair-BaselinePass")
        .usingCommandList(frame.pCommandList)
        .addAttachment(pImage, vk::AttachmentLoadOp::eLoad)
        .addAttachment(pDepthBuffer)
        .execute([&](CommandList* pCmd) -> void
        {
            const struct { glm::mat4 model; HairBuffers buffers; uint32_t strandCount; uint32_t colorMode; } pushData = {
                .model       = glm::yawPitchRoll(0.0f, glm::radians(-90.0f), 0.0f),
                .buffers     = buffers,
                .strandCount = static_cast<uint32_t>(mParams->mStrandCount),
                .colorMode   = std::to_underlying(mParams->mColorMode),
            };
            pCmd->bindPipeline(mPipeline.get());
            pCmd->pushData(pushData);
            pCmd->drawMeshTasks(mParams->mHairGeometry->taskGroupSizeX);
        });
}
