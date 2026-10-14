#pragma once

#include "Hair/HairBuffers.hpp"
#include "Hair/HairRendererParams.hpp"
#include "VulkanRHI/VulkanRHI.hpp"

class BaselinePass
{
public:
    explicit BaselinePass(HairRendererParams* pParams, VulkanRHI* pRHI);

    void render(const Frame& frame, const HairBuffers& buffers, Texture* pDepthBuffer) const;

private:
    VulkanRHI*                        mVulkanRHI;
    HairRendererParams*               mParams;
    std::unique_ptr<GraphicsPipeline> mPipeline;
};
