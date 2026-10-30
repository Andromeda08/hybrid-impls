#pragma once

float3 kajiyaKay(float3 color, float p, float3 T, float3 L, float3 V)
{
    float cosTL = dot(T, L);
    float sinTL = sqrt(max(0.0, 1.0 - cosTL * cosTL));

    float3 H = normalize(L + V);
    float cosTH = dot(T, H);
    float sinTH = sqrt(max(0.0, 1.0 - cosTH * cosTH));

    float3 d = color * sinTL;
    float3 s = color * pow(max(sinTH, 0.0), p);

    return d + s;
}
