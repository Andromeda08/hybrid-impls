#language slang 2026

static const uint groupSize = 64;

struct PushConstants
{
    uint*  counter;
    uint3* indirectArgs;
    uint   maxGroups;
};

[[vk::push_constant]]
ConstantBuffer<PushConstants> pc;

[shader("compute")]
[numthreads(1, 1, 1)]
void main()
{
    uint count = pc.counter[0];
    uint gx    = min((count + groupSize - 1) / groupSize, pc.maxGroups);
    pc.indirectArgs[0] = uint3(gx, 1, 1);
}
