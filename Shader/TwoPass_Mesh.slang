#language slang 2026

#include "lib/Shared.slang"

struct PushConstants
{
    float4x4             model;
    HairVertex*          vertices;
    HairAttributes*      attributes;
    HairStrandDesc*      strandDescriptions;
    GPUCameraData*       camera;
    ScreenSpaceTriangle* trangles;
    uint*                triangleCount;
    uint                 strandCount;
    uint                 colorMode;
    float2               viewport;
    float                threshold;

    HairStrandDesc getStrandDescription(uint globalStrandIndex)
    {
        // A single buffer is used for all hair models, we need the global index for that buffer.
        return strandDescriptions[globalStrandIndex];
    }
};
[[vk::push_constant]] PushConstants pc;

float3 getVertexColor(uint strandId, uint strandletId, uint vertexId)
{
    switch (ColorMode(pc.colorMode))
    {
        case ColorMode::Strand:     return getColor(hash(strandId));
        case ColorMode::Strandlet:  return getColor(hash(strandId) + strandletId);
        case ColorMode::Quad:       return getColor(vertexId);
        default:                    return pc.attributes[vertexId].color;
    }
}

groupshared TaskPayload taskOut;

[shader("amplification")]
[numthreads(nbl_Hair_Workgroup_Size, 1, 1)]
[require(spvGroupNonUniformArithmetic, spvGroupNonUniformShuffle)]
void taskMain(
    uint  threadIndex : SV_GroupIndex,
    uint3 groupID     : SV_GroupID
)
{
    const uint baseID = groupID.x * nbl_Hair_Workgroup_Size;

    const uint localStrandIndex  = threadIndex;
    const uint globalStrandIndex = baseID + localStrandIndex;

    uint strandletCount = (globalStrandIndex < pc.strandCount)
        ? pc.getStrandDescription(globalStrandIndex).strandletCount
        : 0;

    uint strandWorkgroupOffset = WavePrefixSum(strandletCount);

    taskOut.deltaID[threadIndex] = uint8_t(strandWorkgroupOffset);
    taskOut.baseID = baseID;

    GroupMemoryBarrierWithGroupSync();

    uint sumStrandletCount = WaveReadLaneAt(strandWorkgroupOffset + strandletCount, nbl_Hair_Workgroup_Size - 1);
    DispatchMesh(sumStrandletCount, 1, 1, taskOut);
}

[shader("mesh")]
[numthreads(nbl_Hair_Workgroup_Size, 1, 1)]
[outputtopology("triangle")]
void meshMain(
    // Workgroup
    uint  laneId   : SV_GroupIndex,
    uint3 groupId  : SV_GroupID,
    // Mesh Shader
    in  payload  TaskPayload inPayload,
    out vertices MeshVertex  verts[64],
    out indices  uint3       tris[64]
)
{
    uint strandId, strandletId;
    resolveMeshTask(groupId.x, inPayload.baseID, inPayload.deltaID, strandId, strandletId);

    // Thread [Strand] information
    HairStrandDesc  strandDesc = pc.getStrandDescription(strandId);

    // Thread [Strandlet] information
    uint strandletFirstVertex = strandletId * nbl_StrandletStride;
    uint strandletVertices    = getStrandletVertexCount(strandletId, strandDesc.vertexCount);

    // Thread specific offset into hair vertex buffers computed from strand & strandlet
    uint vertexBufferOffset = strandDesc.firstVertex + strandletFirstVertex + laneId;

    // Calculate output parameters
    uint quadCount = strandletVertices - 1;
    SetMeshOutputCounts(strandletVertices * 2, quadCount * 2); // (Vtx, Prim)

    // Early return for shorter than wg size strandlets
    if (laneId >= strandletVertices)
    {
        return;
    }

    float4 strandVertex = float4(pc.vertices[vertexBufferOffset].position, 1.0);

    // Tangent is computed from the previous vertex at the last vertex of a strand.
    bool isStrandEnd = strandletFirstVertex + laneId == strandDesc.vertexCount - 1;
    float3 T = normalize(isStrandEnd
        ? pc.vertices[vertexBufferOffset - 1].position - strandVertex.xyz
        : strandVertex.xyz - pc.vertices[vertexBufferOffset + 1].position);

    // Camera facing quad extrusion
    float4 wStrandPos   = mul(pc.model, strandVertex);
    float3 worldTangent = normalize(mul((float3x3) pc.model, T));

    float3 V = normalize(pc.camera.eye.xyz - wStrandPos.xyz);
    float4 offsetVector = float4(normalize(cross(V, worldTangent)), 0.0);
    float4 wOffsetPos = wStrandPos + offsetVector * pc.attributes[vertexBufferOffset].thickness;

    float4 A = (laneId % 2 == 0) ? wStrandPos : wOffsetPos;
    float4 B = (laneId % 2 == 0) ? wOffsetPos : wStrandPos;

    const var  VP        = pc.camera.getViewProj();
    const uint outOffset = laneId * 2;

    // A --- B
    // |  \  |
    // a --- b
    float4 clipA     = mul(VP, A);
    float4 clipNextA = float4(0.0);
    float4 clipB     = mul(VP, B);
    float4 clipNextB = float4(0.0);

    bool isSmallTri0 = false;
    bool isSmallTri1 = false;
    if (!isStrandEnd)
    {
        const uint nextOffset = vertexBufferOffset + 1;
        float4 wNextPos       = mul(pc.model, float4(pc.vertices[nextOffset].position, 1.0));
        float4 wNextOffsetPos = wNextPos + offsetVector * pc.attributes[nextOffset].thickness;

        float4 a = ((laneId + 1) % 2 == 0) ? wNextPos       : wNextOffsetPos;
        float4 b = ((laneId + 1) % 2 == 0) ? wNextOffsetPos : wNextPos;
        clipNextA = mul(VP, a);
        clipNextB = mul(VP, b);

        float2 sA = toScreen(clipA,     pc.viewport);
        float2 sB = toScreen(clipB,     pc.viewport);
        float2 sa = toScreen(clipNextA, pc.viewport);
        float2 sb = toScreen(clipNextB, pc.viewport);

        isSmallTri0 = isSmallTriangle(sA, sB, sa, pc.threshold);
        isSmallTri1 = isSmallTriangle(sB, sb, sa, pc.threshold);

        if (isSmallTri0)
        {

        }
        if (isSmallTri1)
        {

        }
    }

    float3 color = getVertexColor(strandId, strandletId, vertexBufferOffset);
    verts[outOffset + 0] = { clipA, A.xyz, worldTangent, color };
    verts[outOffset + 1] = { clipB, B.xyz, worldTangent, color };

    if (laneId < quadCount)
    {
        const uint triOffset = laneId * 2;
        tris[triOffset + 0] = isSmallTri0 ? uint3(0) : uint3(0, 1, 2) + outOffset;
        tris[triOffset + 1] = isSmallTri1 ? uint3(0) : uint3(1, 3, 2) + outOffset;
    }
}

[shader("fragment")]
float4 fragMain(MeshVertex input) : SV_Target
{
    const float3 lightPos = float3(-75.0, 125.0, 50.0);

    float3 T = normalize(input.worldTangent);
    float3 L = normalize(lightPos.xyz - input.worldPosition);
    float3 V = normalize(pc.camera.eye.xyz - input.worldPosition);

    float3 diffuse  = float3(0.32549, 0.23921, 0.20784);
    float3 specular = float3(0.41568, 0.30588, 0.21960);

    float3 color = pc.colorMode == ColorMode::Model
        ? kajiyaKay(input.color, input.color, 16.0, T, L, V)
        : input.color;

    return float4(color, 1.0);
}

