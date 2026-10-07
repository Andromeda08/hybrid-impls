#pragma once

#include <array>
#include <ranges>
#include <imgui.h>
#include <glm/glm.hpp>
#include <glm/gtx/euler_angles.hpp>

#include "Hair/HairBuffers.hpp"
#include "Hair/HairGeometry.hpp"
#include "VulkanRHI/VulkanRHI.hpp"

class BaselinePass
{
public:
    explicit BaselinePass(HairGeometry* pHairGeometry, VulkanRHI* pRHI)
    : mVulkanRHI(pRHI)
    , mHairGeometry(pHairGeometry)
    {
        assert(mHairGeometry && mVulkanRHI);
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

    void render(const Frame& frame, const HairBuffers& buffers, Texture* pDepthBuffer) const
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
                    .strandCount = mRenderNStrands ? mStrandCountOverride : mHairGeometry->getStrandCount(),
                    .colorMode   = mColorMode,
                };
                pCmd->bindPipeline(mPipeline.get());
                pCmd->pushData(pushData);
                pCmd->drawMeshTasks(mHairGeometry->taskGroupSizeX);
            });
    }

    void onDrawUI()
    {
        static constexpr std::array colorModeNames = { "Model", "Strand", "Strandlet", "Quad" };

        ImGui::Begin("Baseline Pass");

        ImGui::SeparatorText("Color Mode");
        if (ImGui::BeginCombo("##ColorModeSelect", colorModeNames[mColorMode]))
        {
            for (const auto& [i, name] : std::views::enumerate(colorModeNames))
            {
                const bool isSelected = i == mColorMode;

                if (ImGui::Selectable(name, isSelected))
                {
                    mColorMode = static_cast<uint32_t>(i);
                }
                if (isSelected)
                {
                    ImGui::SetItemDefaultFocus();
                }
            }

            ImGui::EndCombo();
        }

        ImGui::Separator();
        ImGui::Checkbox("Override strand count", &mRenderNStrands);
        ImGui::DragInt("N", &mStrandCountOverride,10, 0, static_cast<int32_t>(mHairGeometry->getStrandCount()));

        ImGui::End();
    }

private:
    VulkanRHI*                        mVulkanRHI;
    HairGeometry*                     mHairGeometry;
    std::unique_ptr<GraphicsPipeline> mPipeline;

    uint32_t mColorMode = 0;

    bool    mRenderNStrands      = false;
    int32_t mStrandCountOverride = 0;
};
