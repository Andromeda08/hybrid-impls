#pragma once

#include <memory>
#include <print>
#include <string>
#include <imgui.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtx/rotate_vector.hpp>
#include <glm/gtx/vector_angle.hpp>
#include <SDL3/SDL_events.h>

#include "VulkanRHI/VulkanRHI.hpp"

struct GPUCameraData
{
    glm::mat4 view;
    glm::mat4 proj;
    glm::mat4 viewProj;
    glm::mat4 viewInverse;
    glm::mat4 projInverse;
    glm::vec4 eye;
    float     nearPlane;
    float     farPlane;
};

class Camera
{
public:
    explicit Camera(const std::string& label) : mLabel(label) {}

    virtual ~Camera() = default;

    virtual void onEvent(const SDL_Event& event) {}

    virtual void onUpdate() {}

    [[nodiscard]] glm::vec4 getEye() const noexcept
    {
        return {  mEye, 1.0f };
    }

    [[nodiscard]] GPUCameraData getGpuData() const noexcept
    {
        return {
            .view        = mView,
            .proj        = mProj,
            .viewProj    = mProj * mView,
            .viewInverse = glm::inverse(mView),
            .projInverse = glm::inverse(mProj),
            .eye         = getEye(),
            .nearPlane   = mNearPlane,
            .farPlane    = mFarPlane,
        };
    };

    [[nodiscard]] const std::string& getName() const noexcept
    {
        return mLabel;
    }

protected:
    glm::mat4   mView      = glm::mat4(1.0f);
    glm::mat4   mProj      = glm::mat4(1.0f);
    glm::vec3   mEye       = glm::vec4(0.0f);
    float       mNearPlane = 0.01f;
    float       mFarPlane  = 4096.0f;
    std::string mLabel;
};

class FlyingCamera final : public Camera
{
public:
    FlyingCamera(glm::ivec2 size, glm::vec3 eye, float near = 0.01f, float far = 4096.0f)
    : Camera("FlyingCamera")
    , mSize({ static_cast<float>(size.x), static_cast<float>(size.y) })
    {
        mEye       = eye;
        mNearPlane = near;
        mFarPlane  = far;
    }

    ~FlyingCamera() override = default;

    void onEvent(const SDL_Event& event) override
    {
        auto* pWindow = SDL_GetWindowFromID(event.button.windowID);
        switch (event.type)
        {
            case SDL_EVENT_KEY_DOWN:
            case SDL_EVENT_KEY_UP: {
                handleKeyEvent(event.key);
                break;
            }
            case SDL_EVENT_MOUSE_BUTTON_DOWN: {
                if (event.button.button == SDL_BUTTON_LEFT)
                {
                    mMouseCaptured = true;
                    SDL_SetWindowRelativeMouseMode(pWindow, true);
                }
                break;
            }
            case SDL_EVENT_MOUSE_BUTTON_UP: {
                if (event.button.button == SDL_BUTTON_LEFT)
                {
                    mMouseCaptured = false;
                    mInputState = {};
                    SDL_SetWindowRelativeMouseMode(pWindow, false);
                    SDL_WarpMouseInWindow(pWindow, mSize.x / 2.0f, mSize.y / 2.0f);
                }
                break;
            }
            case SDL_EVENT_MOUSE_MOTION: {
                if (mMouseCaptured)
                {
                    handleMouseEvent(event.motion);
                }
                break;
            }
            case SDL_EVENT_WINDOW_FOCUS_LOST: {
                mInputState = {};
                mMouseCaptured = false;
                SDL_SetWindowRelativeMouseMode(pWindow, false);
                break;
            }
            default:
                break;
        }
    }

    void onUpdate() override
    {
        const glm::vec3 right = glm::normalize(glm::cross(mOrientation, mUp));

        if (mInputState.forward)
        {
            mEye += mSpeed * mOrientation;
        }
        if (mInputState.backward)
        {
            mEye -= mSpeed * mOrientation;
        }
        if (mInputState.left)
        {
            mEye -= mSpeed * right;
        }
        if (mInputState.right)
        {
            mEye += mSpeed * right;
        }
        if (mInputState.up)
        {
            mEye += (mSpeed / 2.0f) * mUp;
        }
        if (mInputState.down)
        {
            mEye -= (mSpeed / 2.0f) * mUp;
        }

        const float aspect = mSize.x / mSize.y;
        mView = glm::lookAt(mEye, mEye + mOrientation, mUp);
        mProj = glm::perspective(glm::radians(75.0f), aspect, mNearPlane, mFarPlane);
        mProj[1][1] *= -1.0f;
    }

private:
    void handleKeyEvent(const SDL_KeyboardEvent& keyEvent) noexcept
    {
        const bool pressed = (keyEvent.type == SDL_EVENT_KEY_DOWN);

        switch (keyEvent.scancode)
        {
            case SDL_SCANCODE_W: {
                mInputState.forward = pressed;
                break;
            }
            case SDL_SCANCODE_S: {
                mInputState.backward = pressed;
                break;
            }
            case SDL_SCANCODE_A: {
                mInputState.left = pressed;
                break;
            }
            case SDL_SCANCODE_D: {
                mInputState.right = pressed;
                break;
            }
            case SDL_SCANCODE_SPACE: {
                mInputState.up = pressed;
                break;
            }
            case SDL_SCANCODE_LCTRL: {
                mInputState.down = pressed;
                break;
            }
            default: {
                break;
            }
        }
    }

    void handleMouseEvent(const SDL_MouseMotionEvent& motionEvent) noexcept
    {
        const float rotX = mSensitivity * motionEvent.yrel / mSize.y;
        const float rotY = mSensitivity * motionEvent.xrel / mSize.x;

        const glm::vec3 newOrientation = glm::rotate(
            mOrientation,
            glm::radians(-rotX),
            glm::normalize(glm::cross(mOrientation, mUp))
        );

        if (glm::abs(glm::angle(newOrientation, mUp) - glm::radians(90.0f)) <= glm::radians(85.0f))
        {
            mOrientation = newOrientation;
        }

        mOrientation = glm::rotate(mOrientation, glm::radians(-rotY), mUp);
    }

    glm::vec2       mSize;
    float           mSpeed       = 0.25f;
    float           mSensitivity = 50.0f;
    glm::vec3       mUp          = { 0, 1,  0 };
    glm::vec3       mOrientation = { 0, 0, -1 };

    bool mMouseCaptured = false;
    struct KeyState {
        bool forward  = false;
        bool backward = false;
        bool left     = false;
        bool right    = false;
        bool up       = false;
        bool down     = false;
    } mInputState;
};

class OrbitCamera final : public Camera
{
public:
    explicit OrbitCamera(const float initDistance = 35.0f)
    : Camera("OrbitCamera")
    , mRadius(initDistance)
    {
    }

    ~OrbitCamera() override = default;

    void onEvent(const SDL_Event& event) override
    {
        switch (event.type)
        {
            case SDL_EVENT_MOUSE_BUTTON_DOWN: {
                if (event.button.button == SDL_BUTTON_LEFT)
                {
                    mDragging = true;
                }
                break;
            }
            case SDL_EVENT_MOUSE_BUTTON_UP: {
                if (event.button.button == SDL_BUTTON_LEFT)
                {
                    mDragging = false;
                }
                break;
            }
            case SDL_EVENT_MOUSE_MOTION: {
                if (mDragging)
                {
                    orbit(event.motion.yrel * mOrbitSpeed, event.motion.xrel * mOrbitSpeed);
                }
                break;
            }
            case SDL_EVENT_MOUSE_WHEEL: {
                zoom(-event.wheel.y * mZoomSpeed);
                break;
            }
            default: {
                break;
            }
        }
    }

    void onUpdate() override
    {
        mEye = {
            mRadius * glm::sin(mTheta) * glm::cos(mPhi),
            mRadius * glm::cos(mTheta),
            mRadius * glm::sin(mTheta) * glm::sin(mPhi),
        };

        mView = glm::lookAt(mEye, glm::vec3(0.0f), glm::vec3(0.0f, 1.0f, 0.0f));
        mProj = glm::perspective(glm::radians(75.0f), 16.0f / 9.0f, 0.1f, 4096.0f);
        mProj[1][1] *= -1;
    }

private:
    void orbit(const float dTheta, const float dPhi) noexcept
    {
        constexpr float epsilon = 1e-4f;
        mTheta                  = glm::clamp(mTheta + dTheta, epsilon, glm::pi<float>() - epsilon);
        mPhi                    += dPhi;
    }

    void zoom(const float delta) noexcept
    {
        mRadius = glm::max(0.01f, mRadius + delta);
    }

    float                    mPhi           = 0.0f;
    float                    mTheta         = glm::half_pi<float>();
    float                    mRadius        = 5.0f;
    bool                     mDragging      = false;
    float                    mOrbitSpeed    = 0.005f;
    float                    mZoomSpeed     = 1.5f;
};

class CameraSystem
{
public:
    explicit CameraSystem(VulkanRHI* rhi)
    : mVulkanRHI(rhi)
    {
        for (size_t i = 0; i < mBuffers.size(); i++)
        {
            mBuffers[i] = Buffer::Builder()
                .setSize(sizeof(GPUCameraData))
                .setType(BufferType::Uniform)
                .setLabel(fmt::format("CameraSystem_DataBuffer_{}", i))
                .create(mVulkanRHI->getDevice());
        }
    }

    template <class T, class... Args>
    requires std::derived_from<T, Camera>
    T* addCamera(Args&&... args)
    {
        mCameras.push_back(std::make_unique<T>(std::forward<Args>(args)...));
        if (mActiveCamera == -1)
        {
            mActiveCamera = static_cast<int32_t>(mCameras.size()) - 1;
        }
        return static_cast<T*>(mCameras.back().get());
    }

    void onEvent(const SDL_Event& event) const
    {
        if (auto* camera = getActiveCamera())
        {
            camera->onEvent(event);
        }
    }

    void onUpdate(const Frame& frame) const
    {
        if (auto* camera = getActiveCamera())
        {
            camera->onUpdate();
            const auto gpuCameraData = camera->getGpuData();
            mBuffers[frame.currentFrame]->setData(&gpuCameraData, sizeof(GPUCameraData), 0);
        }
    }

    void onDrawUI()
    {
        ImGui::Begin("Cameras");

        const auto* activeCamera = getActiveCamera();
        const std::string activeName = activeCamera ? activeCamera->getName() : "None";

        ImGui::SeparatorText("Select Camera");
        if (ImGui::BeginCombo("##CameraSelect", activeName.c_str()))
        {
            for (const auto& [i, camera] : std::views::enumerate(mCameras))
            {
                const std::string& name = camera->getName();
                const bool isSelected = i == mActiveCamera;

                if (ImGui::Selectable(name.c_str(), isSelected))
                {
                    setActiveCamera(static_cast<int32_t>(i));
                }
                if (isSelected)
                {
                    ImGui::SetItemDefaultFocus();
                }
            }

            ImGui::EndCombo();
        }

        ImGui::End();
    }

    void setActiveCamera(const int32_t index)
    {
        mActiveCamera = index;
    }

    [[nodiscard]] Camera* getActiveCamera() const
    {
        return mActiveCamera == -1 ? nullptr : mCameras[mActiveCamera].get();
    }

    [[nodiscard]] Buffer* getBuffer(const uint64_t frameIndex) const
    {
        return mBuffers[frameIndex].get();
    }

private:
    VulkanRHI*                              mVulkanRHI;
    PerFrameArray<std::unique_ptr<Buffer>>  mBuffers;

    std::vector<std::unique_ptr<Camera>>    mCameras;
    int32_t                                 mActiveCamera = -1;
};
