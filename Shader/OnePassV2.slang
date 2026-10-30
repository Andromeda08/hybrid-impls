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
    RWTexture2D<uint64_t>.Handle depthColor;    // Per pixel: [63:32] depth bits, [29:0] RGB10, cleared to ~0
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

static const uint64_t EmptyPixel = ~0ull;

struct TaskPayload
{
    // Strand processed by this workgroup
    uint strandID;

    // Per mesh workgroup: strandlet to draw
    uint emitted[cfg::MaxStrandletCount];

    // Per mesh workgroup: bit i set => segment i was already rasterized as a line, mesh shader skips that quad
    uint lineMask[cfg::MaxStrandletCount];
};

struct ProjectedVertex
{
    float3 screen; // [x, y] pixels [z] depth
    float  width;  // Strand width in pixels
    float3 world;   // World position for shading
    float3 tangent; // World tangent for shading

    static ProjectedVertex make(uint vertexIndex, float3 T, float4x4 VP)
    {
        float4 wPos = mul(pc.model, float4(pc.pVertices[vertexIndex].position, 1.0));
        float3 wT   = normalize(mul((float3x3) pc.model, T));
        float3 V    = pc.pCamera.getViewDir(wPos.xyz);
        // float4 wExt = wPos + float4(normalize(cross(V, wT)), 0.0) * pc.pAttributes[vertexIndex].thickness;

        float4 clip    = mul(VP, wPos);
        // float4 clipExt = mul(VP, wExt);

        float2 s    = pc.toScreen(clip);
        // float2 sExt = pc.toScreen(clipExt);

        ProjectedVertex v;
        v.screen = float3(s, clip.z / clip.w);
        // v.width  = length(sExt - s);
        v.width   = pc.pAttributes[vertexIndex].thickness * abs(pc.pCamera.proj[1][1]) * 0.5 * pc.viewportSize.y / clip.w;
        v.world   = wPos.xyz;
        v.tangent = wT;
        return v;
    }
};

uint packColor(float3 color)
{
    uint3 q = uint3(saturate(color) * 1023.0 + 0.5);
    return q.r | (q.g << 10) | (q.b << 20);
}

float3 unpackColor(uint packed)
{
    return float3(packed & 0x3FF, (packed >> 10) & 0x3FF, (packed >> 20) & 0x3FF) / 1023.0;
}

bool classifyLine(ProjectedVertex a, ProjectedVertex b)
{
    return max(a.width, b.width) < pc.lineThreshold;
}

// DDA, half-open [a, b) so shared endpoints aren't written twice
// Depth test and color write are one 64-bit atomic min with no return value, nothing to wait on
void rasterizeLine(ProjectedVertex a, ProjectedVertex b, float3 colorA, float3 colorB)
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

        uint64_t packed = (uint64_t(asuint(p.z)) << 32) | uint64_t(packColor(lerp(colorA, colorB, t)));
        InterlockedMin(pc.depthColor[coord], packed);
    }
}

groupshared TaskPayload taskOut;

// Hybrid Task shader variant with line rasterization
// =============================================================
// - 1 workgroup = 1 strand, 1 lane = 1 vertex of the current strandlet
// - Strandlets are processed one after another
// - Thin segments are rasterized as lines directly from the task shader
// - Strandlets with at least one remaining quad are emitted to the mesh shader with a skip mask
[shader("amplification")]
[numthreads(cfg::WorkgroupSize, 1, 1)]
[require(
    spvGroupNonUniformBallot,
    spvGroupNonUniformShuffle,
    spvInt64Atomics,
    spvAbort
)]
void taskShaderMain(
    uint  laneID  : SV_GroupIndex,
    uint3 groupID : SV_GroupID
)
{
    const uint strandID = groupID.x;
    uint emitCount = 0; // Wave-uniform

    if (strandID < pc.strandCount)
    {
        const let strandDesc = pc.pStrandDescriptions[strandID];
        const let VP         = pc.pCamera.getViewProj();
        const let strandLast = strandDesc.firstVertex + strandDesc.vertexCount - 1;

        // =========[ Strandlet Loop ]========
        for (uint strandletID = 0; strandletID < strandDesc.strandletCount; strandletID++)
        {
            const let firstVertex  = strandDesc.getStrandletFirstVertex(strandletID);
            const let vertexCount  = strandDesc.getStrandletVertexCount(strandletID);
            const let segmentCount = vertexCount - 1;

            // Lanes past the strandlet clamp to its last vertex, they stay active for the wave ops
            const uint vertexOffset = firstVertex + min(laneID, vertexCount - 1);

            // Forward difference, backward on the last vertex of the strand
            const bool   isStrandEnd = vertexOffset == strandLast;
            const float3 T = normalize(isStrandEnd
                ? pc.pVertices[vertexOffset - 1].position - pc.pVertices[vertexOffset].position
                : pc.pVertices[vertexOffset].position     - pc.pVertices[vertexOffset + 1].position
            );

            // Each vertex projected and shaded once, segment end comes from the next lane
            const let    V0       = ProjectedVertex::make(vertexOffset, T, VP);
            const float3 L        = normalize(cfg::TestLightPos - V0.world);
            const float3 V        = pc.pCamera.getViewDir(V0.world);
            const float3 shaded   = kajiyaKay(pc.pAttributes[vertexOffset].color, 16.0, V0.tangent, L, V);
            const uint   nextLane = min(laneID + 1, cfg::WorkgroupSize - 1);

            ProjectedVertex V1;
            V1.screen = WaveReadLaneAt(V0.screen, nextLane);
            V1.width  = WaveReadLaneAt(V0.width,  nextLane);
            V1.world   = WaveReadLaneAt(V0.world,   nextLane);
            V1.tangent = WaveReadLaneAt(V0.tangent, nextLane);
            const float3 shadedNext = WaveReadLaneAt(shaded, nextLane);

            const bool isLine = laneID < segmentCount && classifyLine(V0, V1);
            if (isLine)
            {
                rasterizeLine(V0, V1, shaded, shadedNext);
            }

            // Emit to mesh stage only if some quads remain
            const uint mask     = WaveActiveBallot(isLine).x;
            const uint fullMask = (1u << segmentCount) - 1u;
            if (mask != fullMask)
            {
                if (laneID == 0)
                {
                    taskOut.emitted[emitCount]  = strandletID;
                    taskOut.lineMask[emitCount] = mask;
                }
                emitCount++;
            }
        }
    }

    taskOut.strandID = strandID;
    DispatchMesh(emitCount, 1, 1, taskOut);
}

struct MeshVertex
{
    float4 position      : SV_Position;
    float3 worldPosition : POSITION;
    float3 worldTangent  : TANGENT;
    float3 color         : COLOR;
};

[shader("mesh")]
[outputtopology("triangle")]
[numthreads(cfg::WorkgroupSize, 1, 1)]
[require(spvAbort)]
void meshMain(
    uint  laneID  : SV_GroupIndex,
    uint3 groupID : SV_GroupID,
    in payload   TaskPayload taskIn,
    out vertices MeshVertex  verts[cfg::WorkgroupSize * 2],
    out indices  uint3       tris[(cfg::WorkgroupSize - 1) * 2]
)
{
    const uint strandletID = taskIn.emitted[groupID.x];
    const uint lineMask    = taskIn.lineMask[groupID.x];

    const let strandDesc   = pc.pStrandDescriptions[taskIn.strandID];
    const let firstVertex  = strandDesc.getStrandletFirstVertex(strandletID);
    const let vertexCount  = strandDesc.getStrandletVertexCount(strandletID);
    const let segmentCount = vertexCount - 1;

    // Quads left after line rasterization, compacted so no degenerate primitives are emitted
    const uint quadMask  = ~lineMask & ((1u << segmentCount) - 1u);
    const uint quadCount = countbits(quadMask);

    SetMeshOutputCounts(vertexCount * 2, quadCount * 2); // (Vtx, Prim)

    if (laneID >= vertexCount) { return; }

    const uint vertexOffset = firstVertex + laneID;

    // Forward difference, backward on the last vertex of the strand
    const bool   isStrandEnd = vertexOffset == strandDesc.firstVertex + strandDesc.vertexCount - 1;
    const float3 T = normalize(isStrandEnd
        ? pc.pVertices[vertexOffset - 1].position - pc.pVertices[vertexOffset].position
        : pc.pVertices[vertexOffset].position     - pc.pVertices[vertexOffset + 1].position
    );

    // Camera facing quad extrusion
    const float4 wPos = mul(pc.model, float4(pc.pVertices[vertexOffset].position, 1.0));
    const float3 wT   = normalize(mul((float3x3) pc.model, T));
    const float3 V    = pc.pCamera.getViewDir(wPos.xyz);
    const float4 wExt = wPos + float4(normalize(cross(V, wT)), 0.0) * pc.pAttributes[vertexOffset].thickness;

    const let    VP    = pc.pCamera.getViewProj();
    const float3 color = pc.pAttributes[vertexOffset].color;

    // 2i on the strand, 2i + 1 offset
    verts[laneID * 2 + 0] = { mul(VP, wPos), wPos.xyz, wT, color };
    verts[laneID * 2 + 1] = { mul(VP, wExt), wExt.xyz, wT, color };

    // S_i ---- O_i
    //  |    \   |
    // S_i+1 -- O_i+1
    if (((quadMask >> laneID) & 1u) != 0)
    {
        const uint quadIndex = countbits(quadMask & ((1u << laneID) - 1u));
        const uint v         = laneID * 2;
        tris[quadIndex * 2 + 0] = uint3(v + 0, v + 1, v + 2);
        tris[quadIndex * 2 + 1] = uint3(v + 1, v + 3, v + 2);
    }
}

[shader("fragment")]
[earlydepthstencil]
void fragMain(MeshVertex input)
{
    const int2 coord = int2(input.position.xy);
    const uint zBits = asuint(input.position.z);

    // if (uint(pc.depthColor[coord] >> 32) < zBits) { return; }

    const float3 T     = normalize(input.worldTangent);
    const float3 L     = normalize(cfg::TestLightPos - input.worldPosition);
    const float3 V     = pc.pCamera.getViewDir(input.worldPosition);
    const float3 color = kajiyaKay(input.color, 16.0, T, L, V);

    const uint64_t packed = (uint64_t(zBits) << 32) | uint64_t(packColor(color));
    InterlockedMin(pc.depthColor[coord], packed);
}

// Resolve packed depth + color into the render target
// =============================================================
[shader("compute")]
[numthreads(8, 8, 1)]
void resolveMain(uint3 threadID : SV_DispatchThreadID)
{
    const int2 coord = int2(threadID.xy);
    if (any(coord >= int2(pc.viewportSize))) { return; }

    const uint64_t packed = pc.depthColor[coord];

    pc.renderTarget[coord] = packed == EmptyPixel
        ? float4(0.0, 0.0, 0.0, 1.0)
        : float4(unpackColor(uint(packed & 0xFFFFFFFFull)), 1.0);
}
