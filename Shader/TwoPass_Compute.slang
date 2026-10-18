#language slang 2026

#include "lib/Shared.slang"

struct PushConstants
{
    HairVertex*                 vertices;
    HairAttributes*             attributes;
    HairStrandDesc*             strandDescriptions;
    GPUCameraData*              camera;
    ScreenSpaceTriangle*        triangles;
    uint*                       triangleCount;
    float2                      viewport;
    RWTexture2D<float4>.Handle  renderTarget;
    RWTexture2D<uint>.Handle    depthBuffer;
    uint                        colorMode;

    BBoxInt2 getScissor()
    {
        return BBoxInt2::fromScissor(0, 0, uint(pc.viewport.x), uint(pc.viewport.y));
    }
};
[[vk::push_constant]] PushConstants pc;

[shader("compute"), numthreads(64, 1, 1)]
void computeMain(uint3 threadId : SV_DispatchThreadID)
{
    uint triIndex = threadId.x;
    if (triIndex >= pc.triangleCount[0]) { return; }

    var input    = pc.triangles[triIndex];
    var triangle = Triangle::fromInput(input);

    BBoxInt2 clipped = triangle.bounds.getClippedBy(pc.getScissor());
    if (clipped.isEmpty()) { return; }

    // Loop over pixels
    for (int y = clipped.bboxMin.y; y <= clipped.bboxMax.y; y++)
    {
        for (int x = clipped.bboxMin.x; x <= clipped.bboxMax.x; x++)
        {
            var coord = int2(x, y);
            var pixel = float2(x, y) + 0.5;
            var bary  = triangle.bary(pixel);
            if (any(bary < 0.0f)) { continue; }

            float3 interp    = (bary.x * input.v0) + (bary.y * input.v1) + (bary.z * input.v2);
            float2 screenPos = interp.xy;
            float  depth     = interp.z;

            // Depth test
            uint depthBits = asuint(depth);
            if (depthBits >= pc.depthBuffer[coord]) { continue; }

            float2 ndc = (pixel / pc.viewport) * 2.0 - 1.0;
            float4 viewPos = mul(pc.camera.projInverse, float4(ndc, depth, 1.0));
            float3 worldPos = mul(pc.camera.viewInverse, viewPos / viewPos.w).xyz;

            // float3 worldPos = bary.x * w0 + bary.y * w1 + bary.z * w2;

            // BSDF, width default light at index 0.
            const float3 lightPos = float3(-75.0, 125.0, 50.0);
            float3 T = normalize(input.tangent);
            float3 L = normalize(lightPos.xyz - worldPos);
            float3 V = normalize(pc.camera.eye.xyz - worldPos);

            float3 color = pc.colorMode == ColorMode::Model
                ? kajiyaKay(input.color, input.color, 16.0, T, L, V)
                : pc.colorMode == ColorMode::Compute
                    ? getColor(triIndex)
                    : input.color;

            // Write
            uint prevBits;
            InterlockedMin(pc.depthBuffer[coord], depthBits, prevBits);
            if (depthBits < prevBits)
            {
                pc.renderTarget[coord] = float4(color, 1.0);
            }
        }
    }
}
