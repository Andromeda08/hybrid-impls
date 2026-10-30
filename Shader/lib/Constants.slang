#pragma once

namespace cfg
{
    static const uint WorkgroupSize     = 32;
    static const uint StrandletStride   = WorkgroupSize - 1;
    static const uint MaxStrandletCount = 4;

    static const float3 TestLightPos = float3(-75.0, 125.0, 50.0);
}