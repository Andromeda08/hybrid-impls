#pragma once

#include <array>
#include <cstdint>
#include <string_view>

#include <imgui.h>
#include <ranges>

#include "HairGeometry.hpp"

enum class HairVisMode : uint32_t
{
    Model       = 0,
    Strand      = 1,
    Strandlet   = 2,
    Quad        = 3,
    Compute     = 4,
};

[[nodiscard]] constexpr std::string_view toString(const HairVisMode e) noexcept
{
    using enum HairVisMode;
    switch (e)
    {
        case Model:     return "Model";
        case Strand:    return "Strand";
        case Strandlet: return "Strandlet";
        case Quad:      return "Quad";
        case Compute:   return "Compute";
    }
    std::unreachable();
}

struct HairRendererParams
{
    uint32_t        mActiveRenderer = 0;
    HairGeometry*   mHairGeometry = nullptr;

    // Geometry visualization mode
    HairVisMode     mColorMode      = HairVisMode::Model;

    // Strand count override
    int32_t         mStrandCount    = 0;

    // Thickness override
    bool            mForceThickness = false;
    float           mThickness      = 0.25f;

    // Hybrid renderer classification
    float           mTriThreshold  = 4.0f;
    float           mLineThreshold = 0.5f;

    explicit HairRendererParams(HairGeometry* pHairGeometry)
    : mHairGeometry(pHairGeometry)
    {
        mStrandCount = static_cast<int32_t>(mHairGeometry->getStrandCount());
    }

    void onDrawUI()
    {
        using enum HairVisMode;
        static constexpr std::array sVisModeNames = {
            toString(Model), toString(Strand), toString(Strandlet), toString(Quad), toString(Compute),
        };

        ImGui::Begin("Hair Renderer");

        static constexpr std::array sRendererNames = { "Baseline", "Two-Pass Hybrid", "Combined Hybrid", "Combined Hybrid V2" };
        if (ImGui::BeginCombo("##RendererSelect", sRendererNames[mActiveRenderer]))
        {
            for (auto&& [i, name] : std::views::enumerate(sRendererNames))
            {
                const bool isSelected = mActiveRenderer == i;
                if (ImGui::Selectable(name, isSelected))
                {
                    mActiveRenderer = i;
                }
                if (isSelected)
                {
                    ImGui::SetItemDefaultFocus();
                }
            }
            ImGui::EndCombo();
        }

        ImGui::SeparatorText("Hair Stats");
        ImGui::Text("Vertices   : %u",   mHairGeometry->getVertexCount());
        ImGui::Text("Strands    : %u",   mHairGeometry->getStrandCount());
        ImGui::Text("Strandlets : %llu", mHairGeometry->strandlets.size());

        ImGui::SeparatorText("Strand Count");

        if (ImGui::Button("10"))
        {
            mStrandCount = 10;
        }
        ImGui::SameLine();
        if (ImGui::Button("Max"))
        {
            mStrandCount = static_cast<int32_t>(mHairGeometry->getStrandCount());
        }

        ImGui::DragInt("N", &mStrandCount, 10, 0, static_cast<int32_t>(mHairGeometry->getStrandCount()));

        ImGui::SeparatorText("Color");
        if (ImGui::BeginCombo("##ColorSelect", sVisModeNames[std::to_underlying(mColorMode)].data()))
        {
            for (auto&& [i, name] : std::views::enumerate(sVisModeNames))
            {
                const bool isSelected = std::to_underlying(mColorMode) == i;
                if (ImGui::Selectable(name.data(), isSelected))
                {
                    mColorMode = static_cast<HairVisMode>(i);
                }
                if (isSelected)
                {
                    ImGui::SetItemDefaultFocus();
                }
            }
            ImGui::EndCombo();
        }

        ImGui::SeparatorText("Thickness");
        ImGui::Checkbox("Force", &mForceThickness);
        ImGui::SameLine();
        ImGui::DragFloat("##Thickness", &mThickness, 0.05f, 0.0f, 1.0f);

        ImGui::SeparatorText("Triangle Threshold");
        ImGui::DragFloat("Area", &mTriThreshold, 0.05f, 0.0f, 8.0f);

        ImGui::SeparatorText("Line Threshold");
        ImGui::DragFloat("Width", &mLineThreshold, 0.025f, 0.0f, 2.0f);


        ImGui::End();
    }
};