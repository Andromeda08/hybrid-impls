#pragma once

#include <algorithm>
#include "VulkanRHI/VulkanRHI.hpp"

class GPUProfiler
{
public:
    explicit GPUProfiler(VulkanRHI* pRHI)
    : mRHI(pRHI)
    {
        mTimestampPeriod = mRHI->getDevice()->getPhysicalDevice().getProperties2().properties.limits.timestampPeriod;
        for (uint32_t i = 0; i < gFramesInFlight; i++)
        {
            const auto device = mRHI->getDevice()->getHandle();
            mTimestampPools[i] = device.createQueryPool({{}, vk::QueryType::eTimestamp, 2 });
            mMeshPrimPools[i]  = device.createQueryPool({{}, vk::QueryType::eMeshPrimitivesGeneratedEXT, 1 });
        }
    }

    ~GPUProfiler()
    {
        for (uint32_t i = 0; i < gFramesInFlight; i++)
        {
            const auto device = mRHI->getDevice()->getHandle();
            device.destroy(mTimestampPools[i]);
            device.destroy(mMeshPrimPools[i]);
        }
    }

    void beginQuery(const CommandList* pCommandList, const uint32_t currentFrame)
    {
        if (mIsCapturing)
        {
            mCaptureFrame++;
        }

        const auto device = mRHI->getDevice()->getHandle();
        if (mQueriesWritten[currentFrame])
        {
            std::array<uint64_t, 2> ts = {};
            std::ignore = device.getQueryPoolResults(mTimestampPools[currentFrame], 0, 2, sizeof(ts), ts.data(), sizeof(uint64_t), vk::QueryResultFlagBits::e64);

            const uint64_t delta = ts[1] - ts[0];
            const auto     gpuMS = static_cast<float>(static_cast<double>(delta) * mTimestampPeriod * 1e-6);

            uint64_t meshPrims = 0;
            std::ignore = device.getQueryPoolResults(mMeshPrimPools[currentFrame], 0, 1, sizeof(uint64_t), &meshPrims, sizeof(uint64_t), vk::QueryResultFlagBits::e64);

            mQueriesWritten[currentFrame] = false;

            commitStats(gpuMS, meshPrims);
        }

        device.resetQueryPool(mTimestampPools[currentFrame], 0, 2);
        device.resetQueryPool(mMeshPrimPools[currentFrame], 0, 1);

        pCommandList->getHandle().writeTimestamp2(vk::PipelineStageFlagBits2::eTopOfPipe, mTimestampPools[currentFrame], 0);
        pCommandList->getHandle().beginQuery(mMeshPrimPools[currentFrame], 0, {});
    }

    void endQuery(const CommandList* pCommandList, const uint32_t currentFrame)
    {
        pCommandList->getHandle().endQuery(mMeshPrimPools[currentFrame], 0);
        pCommandList->getHandle().writeTimestamp2(vk::PipelineStageFlagBits2::eBottomOfPipe, mTimestampPools[currentFrame], 1);
        mQueriesWritten[currentFrame] = true;
    }

    void startCapture()
    {
        if (mIsCapturing)
        {
            return;
        }

        mIsCapturing  = true;
        mCaptureFrame = 0;
        mGpuMS.fill(0.0f);

        std::println("Starting benchmark of {} frames.", mGpuMS.size());
    }

    [[nodiscard]] uint64_t getCaptureFrame() const
    {
        return mCaptureFrame >= sWarmupFrames
            ? mCaptureFrame - sWarmupFrames
            : 0;
    }

    [[nodiscard]] bool isCapturing() const
    {
        return mIsCapturing;
    }

    constexpr static uint64_t sWarmupFrames  = 200;
    constexpr static uint64_t sCaptureFrames = 2048;
    float    lastGpuMS     = 0.0f;
    uint64_t lastMeshPrims = 0;

private:
    void commitStats(const float gpuMS, const uint64_t meshPrims)
    {
        lastGpuMS = gpuMS;
        lastMeshPrims = meshPrims;

        const auto idx = mCaptureFrame - sWarmupFrames;
        if (mIsCapturing && mCaptureFrame >= sWarmupFrames && idx < sCaptureFrames)
        {
            mGpuMS[idx] = gpuMS;

            if (idx + 1 == sCaptureFrames)
            {
                mIsCapturing = false;
                generateReport();
            }
        }
    }

    void generateReport() const
    {
        std::array<float, sCaptureFrames> sorted = mGpuMS;
        std::ranges::sort(sorted);

        constexpr auto n = sorted.size();

        double sum = 0.0;
        for (const float val : sorted)
        {
            sum += val;
        }
        const double mean = sum / static_cast<double>(n);

        double sqSum = 0.0;
        for (const float val : sorted)
        {
            sqSum += (val - mean) * (val - mean);
        }
        const double stdDev = std::sqrt(sqSum / static_cast<double>(n - 1));

        const float median = (n % 2 == 0)
            ? 0.5f * (sorted[n / 2 - 1] + sorted[n / 2])
            : sorted[n / 2];

        std::println("Benchmark Results:\n\t- Mean   : {}\n\t- Median : {}\n\t- StdDev : {}\n\t- Min    : {}\n\t- Max    : {}",
            static_cast<float>(mean), median, static_cast<float>(stdDev), sorted.front(), sorted.back());
    }

    VulkanRHI*                      mRHI;

    PerFrameArray<vk::QueryPool>    mTimestampPools;
    PerFrameArray<vk::QueryPool>    mMeshPrimPools;
    PerFrameArray<bool>             mQueriesWritten  = {};
    float                           mTimestampPeriod = 0.0f;

    bool                                mIsCapturing  = false;
    uint64_t                            mCaptureFrame = 0;
    std::array<float, sCaptureFrames>   mGpuMS        = {};
};