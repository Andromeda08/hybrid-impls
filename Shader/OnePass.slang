#language slang 2026

#include "lib/Camera.slang"
#include "lib/Constants.slang"
#include "lib/HairTypes.slang"
#include "lib/Shading.slang"

struct PushConstant
{
    HairVertex*                 pVertices;
    HairAttributes*             pAttributes;
    HairStrandDescription*      pStrandDescriptions;
    GPUCameraData*              pCamera;
    RWTexture2D<float4>.Handle  renderTarget;
    RWTexture2D<uint>.Handle    depthBuffer;
    float4x4                    model;
    float2                      viewportSize;
    uint                        strandCount;
    float                       lineThreshold;

    float2 toScreen(float4 clipPos)
    {
        float2 ndc = clipPos.xy / clipPos.w;
        return (ndc * 0.5 + 0.5) * viewportSize;
    }
};
[[vk::push_constant]] PushConstant pc;

struct TaskPayload
{
    // Index of the first strand processed by this warp
    uint baseID;

    // Which strandlet to draw, roughly equivalent to resolveMeshTask()
    uint16_t emitted[cfg::WorkgroupSize * cfg::MaxStrandletCount];

    // Which quads have been already rasterized as lines by the task shader
    // lineMask[groupId.x][laneId] gives yes/no for the current mesh shader thread
    uint lineMask[cfg::WorkgroupSize * cfg::MaxStrandletCount];
};

struct ProjectedVertex
{
    float3 screen; // [x, y] pixels [z] depth
    float  width;  // Strand width in pixels
    float3 world;  // World position for shading

    static ProjectedVertex make(uint vertexIndex, float3 T, float4x4 VP)
    {
        float4 wPos = mul(pc.model, float4(pc.pVertices[vertexIndex].position, 1.0));
        float3 wT   = normalize(mul((float3x3) pc.model, T));
        float3 V    = pc.pCamera.getViewDir(wPos.xyz);
        float4 wExt = wPos + float4(normalize(cross(V, wT)), 0.0) * pc.pAttributes[vertexIndex].thickness;

        float4 clip    = mul(VP, wPos);
        float4 clipExt = mul(VP, wExt);

        float2 s    = pc.toScreen(clip);
        float2 sExt = pc.toScreen(clipExt);

        ProjectedVertex v;
        v.screen = float3(s, clip.z / clip.w);
        v.width  = length(sExt - s);
        v.world  = wPos.xyz;
        return v;
    }
};

bool classifyLine(ProjectedVertex a, ProjectedVertex b)
{
    return max(a.width, b.width) < pc.lineThreshold;
}

void rasterizeLine(ProjectedVertex a, ProjectedVertex b, float3 color)
{
    float2 d     = b.screen.xy - a.screen.xy;
    uint   steps = max(uint(ceil(max(abs(d.x), abs(d.y)))), 1u);

    for (uint i = 0; i < steps; i++)
    {
        float  t     = float(i) / float(steps);
        float3 p     = lerp(a.screen, b.screen, t);
        int2   coord = int2(floor(p.xy));

        if (any(coord < 0) || any(coord >= int2(pc.viewportSize))) { continue; }
        if (p.z < 0.0 || p.z > 1.0) { continue; }

        uint depthBits = asuint(p.z);
        if (depthBits >= pc.depthBuffer[coord]) { continue; }

        uint prevBits;
        InterlockedMin(pc.depthBuffer[coord], depthBits, prevBits);
        if (depthBits < prevBits)
        {
            float3 worldPos = lerp(a.world, b.world, t);
            float3 T        = normalize(b.world - a.world);
            float3 L        = normalize(cfg::TestLightPos - worldPos);
            float3 V        = pc.pCamera.getViewDir(worldPos);

            pc.renderTarget[coord] = float4(kajiyaKay(color, 16.0, T, L, V), 1.0);
        }
    }
}

groupshared TaskPayload taskOut;

// Hybrid Task shader variant with line rasterization
// =============================================================
[shader("amplification")]
[numthreads(cfg::WorkgroupSize, 1, 1)]
[require(
    spvGroupNonUniformArithmetic,
    spvGroupNonUniformShuffle,
    spvAbort
)]
void taskShaderMain(
    uint  groupIndex : SV_GroupIndex,
    uint3 groupID    : SV_GroupID
)
{
    const uint baseID   = groupID.x * cfg::WorkgroupSize;   // First strand processed by this warp
    const uint strandID = baseID + groupIndex;              // warpID * warpSize + laneID gives global strand index

    taskOut.baseID = baseID;

    uint     emitCount                        = 0;
    uint16_t emitIds[cfg::MaxStrandletCount]  = {};
    uint     emitMask[cfg::MaxStrandletCount] = {};

    if (strandID < pc.strandCount)
    {
        const let strandDesc = pc.pStrandDescriptions[strandID];
        const let VP         = pc.pCamera.getViewProj();

        // =========[ Strandlet Loop ]========
        for (uint strandletID = 0; strandletID < strandDesc.strandletCount; strandletID++)
        {
            const let firstVertex  = strandDesc.getStrandletFirstVertex(strandletID);
            const let vertexCount  = strandDesc.getStrandletVertexCount(strandletID);
            const let segmentCount = vertexCount - 1;

            uint mask = 0;

            float3 T0 = normalize(pc.pVertices[firstVertex].position - pc.pVertices[firstVertex + 1].position);
            var    V0 = ProjectedVertex::make(firstVertex, T0, VP);

            // =========[ Segment Loop ]========
            for (uint i = 0; i < segmentCount; i++)
            {
                uint vertexOffset = firstVertex + i + 1;

                bool isStrandEnd = vertexOffset == strandDesc.firstVertex + strandDesc.vertexCount - 1;
                float3 T1 = normalize(isStrandEnd
                    ? pc.pVertices[vertexOffset - 1].position - pc.pVertices[vertexOffset].position
                    : pc.pVertices[vertexOffset].position     - pc.pVertices[vertexOffset + 1].position
                );
                var V1 = ProjectedVertex::make(vertexOffset, T1, VP);

                if (classifyLine(V0, V1))
                {
                    rasterizeLine(V0, V1, pc.pAttributes[vertexOffset - 1].color);
                    mask |= 1u << i;
                }

                V0 = V1;
            }

            const uint fullMask = (1u << segmentCount) - 1u;
            if (mask != fullMask)
            {
                emitIds[emitCount]  = uint16_t((groupIndex << 8) | strandletID);
                emitMask[emitCount] = mask;
                emitCount++;
            }
        }
    }

    const uint offset = WavePrefixSum(emitCount);
    const uint total  = WaveActiveSum(emitCount);

    for (uint k = 0; k < emitCount; k++)
    {
        taskOut.emitted[offset + k]  = emitIds[k];
        taskOut.lineMask[offset + k] = emitMask[k];
    }

    DispatchMesh(total, 1, 1, taskOut);
}

struct DummyVertex { float4 position : SV_Position; };

[shader("mesh")]
[outputtopology("triangle")]
[numthreads(cfg::WorkgroupSize, 1, 1)]
void meshMain(
    in payload   TaskPayload taskIn,
    out vertices DummyVertex verts[1],
    out indices  uint3       tris[1]
)
{
    SetMeshOutputCounts(0, 0);
}

[shader("fragment")]
float4 fragMain() : SV_Target { return float4(0.0); }
